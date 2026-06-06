from __future__ import annotations

from collections.abc import Sequence
import os
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from refiner.cli.run.modes import (
    CloudAttachContext,
    attach_mode_override,
    emit_cloud_followup_commands,
    resolve_launcher_attach_mode,
)
from refiner.cli.ui import stdin_is_interactive, stdout_is_interactive
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudFile,
    CloudFileCompleteRequestItem,
    CloudFileUploadInstruction,
    CloudFileUploadRequestItem,
    CloudFileUploadStatus,
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    MacrodataApiError,
    MacrodataClient,
    StagePayload,
)
from refiner.platform.client.serialize import PreparedPipelinePayload
from refiner.platform.manifest import build_run_manifest, refiner_ref_exists_on_remote
from refiner.launchers.secrets import SecretInput, resolve_env_mapping
from refiner.launchers.secrets import normalize_secret_sources, resolve_secret_sources
from refiner.pipeline.resources import GPU
from refiner.services.discovery import collect_pipeline_services

from refiner.job_urls import build_job_tracking_url
from refiner.launchers.base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline
    from refiner.pipeline.planning import PlannedStage


_FALLBACK_ENV_VAR = "MACRODATA_FALLBACK_TO_LATEST_PYPI"
_CLOUD_FILE_BATCH_SIZE = 100
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _parse_continue_from_job(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("continue_from_job must be non-empty")
    if normalized == "infer":
        return normalized
    if normalized.count(":") > 1:
        raise ValueError("continue_from_job must be UUID, UUID:stage_index, or 'infer'")
    if ":" not in normalized:
        if not _UUID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "continue_from_job must be UUID, UUID:stage_index, or 'infer'"
            )
        return normalized
    job_id, raw_stage_index = normalized.split(":", 1)
    if not job_id.strip():
        raise ValueError("continue_from_job job id must be non-empty")
    normalized_job_id = job_id.strip()
    if not _UUID_PATTERN.fullmatch(normalized_job_id):
        raise ValueError("continue_from_job job id must be a UUID")
    if not raw_stage_index.strip():
        raise ValueError("continue_from_job stage index must be non-empty")
    try:
        stage_index = int(raw_stage_index)
    except ValueError as err:
        raise ValueError("continue_from_job stage index must be an integer") from err
    if stage_index < 0:
        raise ValueError("continue_from_job stage index must be >= 0")
    return f"{normalized_job_id}:{stage_index}"


@dataclass(frozen=True, slots=True)
class CloudLaunchResult:
    job_id: str
    stage_index: int
    status: str
    warnings: list[str]


class CloudLauncher(BaseLauncher):
    """Cloud launcher that submits a compiled run to the cloud controller.

    Args:
        pipeline: Pipeline to execute.
        name: Human-readable run name.
        num_workers: Requested logical worker count for cloud execution.
        cpus_per_worker: Optional requested CPU cores per worker.
        mem_mb_per_worker: Optional requested memory in MB per worker for cloud scheduling.
        gpu: Optional GPU runtime request for cloud scheduling.
        sync_local_dependencies: Whether to include packages detected from the
            local environment in the cloud runtime.
        dependencies: Additional packages to install in the cloud runtime.
            Entries are requirement strings.
        refiner_extras: Optional macrodata-refiner extras to install in the cloud
            runtime, such as "hf" or "video".
        secrets: Optional secret sources mounted into the cloud runtime.
        env: Optional plain environment variables mounted into the cloud runtime.
    """

    def __init__(
        self,
        *,
        pipeline: "RefinerPipeline",
        name: str,
        num_workers: int = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
        sync_local_dependencies: bool = False,
        dependencies: Sequence[str] | None = None,
        refiner_extras: Sequence[str] | None = None,
        secrets: SecretInput | None = None,
        env: dict[str, object | None] | None = None,
        continue_from_job: str | None = None,
        unsafe_continue: bool = False,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            gpu=gpu,
        )
        normalized_continue_from_job = _parse_continue_from_job(continue_from_job)
        if unsafe_continue and normalized_continue_from_job is None:
            raise ValueError("unsafe_continue requires continue_from_job")
        if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
            raise ValueError("mem_mb_per_worker must be > 0")
        self.cpus_per_worker = cpus_per_worker
        self.mem_mb_per_worker = mem_mb_per_worker
        self.sync_local_dependencies = sync_local_dependencies
        self.dependencies = dependencies
        self.refiner_extras = refiner_extras
        self.secrets = normalize_secret_sources(secrets)
        self.env = env
        self.continue_from_job = normalized_continue_from_job
        self.unsafe_continue = unsafe_continue

    @staticmethod
    def _fallback_to_latest_pypi_enabled() -> bool:
        raw = os.environ.get(_FALLBACK_ENV_VAR, "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_cloud_manifest(
        self, *, secret_values: tuple[str, ...]
    ) -> dict[str, object]:
        manifest = build_run_manifest(
            secret_values=secret_values,
            capture_dependencies=self.sync_local_dependencies,
            dependencies=self.dependencies,
            refiner_extras=self.refiner_extras,
        )
        environment = manifest.get("environment")
        if environment is None:
            return manifest
        environment_dict = cast(dict[str, object], environment)
        refiner_ref = environment_dict.get("refiner_ref")
        if not isinstance(refiner_ref, str) or not refiner_ref.strip():
            return manifest
        refiner_ref = refiner_ref.strip()
        if refiner_ref_exists_on_remote(refiner_ref):
            return manifest

        message = (
            f"Refiner ref {refiner_ref!r} is not available on GitHub. "
            "Launch with the latest PyPI version instead?"
        )
        fallback_allowed = self._fallback_to_latest_pypi_enabled()
        interactive = stdin_is_interactive()
        if not fallback_allowed and interactive:
            answer = input(f"{message} [y/N] ")
            fallback_allowed = answer.strip().lower() in {"y", "yes"}
        if fallback_allowed:
            environment_dict["refiner_ref"] = None
            return manifest
        if interactive:
            raise SystemExit("cloud launch aborted")

        raise SystemExit(
            f"{message} Launch aborted before submission. "
            f"Set {_FALLBACK_ENV_VAR}=1 to allow fallback to the latest PyPI version."
        )

    @staticmethod
    def _upload_instructions_by_file(
        instructions: list[CloudFileUploadInstruction],
        *,
        expected_files: set[tuple[str, int]],
    ) -> dict[tuple[str, int], CloudFileUploadInstruction]:
        instructions_by_file: dict[tuple[str, int], CloudFileUploadInstruction] = {}
        for instruction in instructions:
            file_key = (instruction.sha256, instruction.size_bytes)
            if file_key in expected_files:
                instructions_by_file[file_key] = instruction

        missing_files = sorted(expected_files - instructions_by_file.keys())
        if missing_files:
            sha256, size_bytes = missing_files[0]
            raise ValueError(
                "Cloud file upload URL response did not return instructions "
                f"for sha256/size_bytes: {sha256}/{size_bytes}"
            )
        return instructions_by_file

    @staticmethod
    def _upload_stage_payloads(
        *,
        client: MacrodataClient,
        stages: list[PlannedStage],
    ) -> dict[int, CloudFile]:
        serialized_payloads = [
            PreparedPipelinePayload.from_pipeline(stage.pipeline) for stage in stages
        ]
        serialized_by_file = {}
        for serialized in serialized_payloads:
            file_key = (serialized.sha256, serialized.size_bytes)
            serialized_by_file.setdefault(file_key, serialized)

        instructions_by_file: dict[tuple[str, int], CloudFileUploadInstruction] = {}
        serialized_items = list(serialized_by_file.items())
        for index in range(0, len(serialized_items), _CLOUD_FILE_BATCH_SIZE):
            batch = serialized_items[index : index + _CLOUD_FILE_BATCH_SIZE]
            upload_response = client.cloud_create_file_upload_urls(
                files=[
                    CloudFileUploadRequestItem(
                        sha256=serialized.sha256,
                        size_bytes=serialized.size_bytes,
                    )
                    for _, serialized in batch
                ],
                object_ttl_secs=None,
            )
            instructions_by_file.update(
                CloudLauncher._upload_instructions_by_file(
                    upload_response.files,
                    expected_files={file_key for file_key, _ in batch},
                )
            )

        completed_files: list[CloudFileCompleteRequestItem] = []
        for file_key, instruction in instructions_by_file.items():
            if instruction.status is CloudFileUploadStatus.EXISTS:
                continue
            serialized = serialized_by_file[file_key]
            client.cloud_upload_file(
                instruction=instruction,
                payload_bytes=serialized.payload_bytes,
            )
            completed_files.append(
                CloudFileCompleteRequestItem(file_id=instruction.file_id)
            )

        for index in range(0, len(completed_files), _CLOUD_FILE_BATCH_SIZE):
            client.cloud_complete_files(
                files=completed_files[index : index + _CLOUD_FILE_BATCH_SIZE],
                object_ttl_secs=None,
            )

        return {
            stage.index: CloudFile(
                file_id=instructions_by_file[
                    (serialized.sha256, serialized.size_bytes)
                ].file_id
            )
            for stage, serialized in zip(stages, serialized_payloads, strict=True)
        }

    def launch(self) -> CloudLaunchResult:
        try:
            client = MacrodataClient()
        except MacrodataCredentialsError as err:
            raise SystemExit(
                "Launching jobs in the Macrodata cloud requires Macrodata "
                "authentication. Run `macrodata login` or set MACRODATA_API_KEY."
            ) from err
        resolved_secret_sources, secret_values = resolve_secret_sources(self.secrets)
        resolved_env = resolve_env_mapping(self.env) if self.env else None
        stages = self._resolved_stages()
        manifest = self._resolve_cloud_manifest(secret_values=secret_values)
        plan = self._compiled_plan(stages, secret_values=secret_values)
        try:
            pipeline_payloads = self._upload_stage_payloads(
                client=client, stages=stages
            )
            request = CloudRunCreateRequest(
                name=self.name,
                plan=plan,
                stage_payloads=[
                    StagePayload(
                        stage_index=stage.index,
                        pipeline_payload=pipeline_payloads[stage.index],
                        runtime=CloudRuntimeConfig(
                            num_workers=stage.compute.num_workers,
                            cpus_per_worker=stage.compute.cpus_per_worker,
                            mem_mb_per_worker=stage.compute.memory_mb_per_worker,
                            gpu=stage.compute.gpu,
                        ),
                        runtime_services=collect_pipeline_services(stage.pipeline),
                    )
                    for stage in stages
                ],
                manifest=manifest,
                secrets=resolved_secret_sources,
                env=resolved_env,
                continue_from_job=self.continue_from_job,
                unsafe_continue=self.unsafe_continue,
            )
            resp = client.cloud_submit_job(request=request)
        except MacrodataCredentialsError as err:
            raise SystemExit(
                "Your Macrodata API key is invalid. Run `macrodata login` "
                "or set MACRODATA_API_KEY with a valid key."
            ) from err
        except ValueError as err:
            raise SystemExit(str(err)) from err
        except MacrodataApiError as err:
            raise SystemExit(err.message) from err
        tracking_url = build_job_tracking_url(
            client=client,
            job_id=resp.job_id,
            workspace_slug=resp.workspace_slug,
        )
        response_warnings = list(getattr(resp, "warnings", []))
        for warning_message in response_warnings:
            print(f"Warning: {warning_message}", file=sys.stderr)
        context = CloudAttachContext(
            job_id=resp.job_id,
            job_name=self.name,
            tracking_url=tracking_url,
            stage_index=resp.stage_index,
        )
        print(f"Cloud job launched. View job:\n  {tracking_url}")
        attach_mode = resolve_launcher_attach_mode(interactive=stdout_is_interactive())
        if attach_mode == "detach":
            emit_cloud_followup_commands(context=context)
        else:
            try:
                from refiner.cli.run.cloud import attach_to_cloud_job

                attach_rc = attach_to_cloud_job(
                    client=client,
                    job_id=resp.job_id,
                    stage_index_hint=resp.stage_index,
                    force_attach=True,
                )
                if attach_rc != 0 and attach_mode_override() is not None:
                    raise SystemExit(attach_rc)
            except (MacrodataApiError, MacrodataCredentialsError):
                print(
                    "Cloud job submitted, but attach failed. Continue with:",
                    file=sys.stderr,
                )
                emit_cloud_followup_commands(context=context, file=sys.stderr)
        return CloudLaunchResult(
            job_id=resp.job_id,
            stage_index=resp.stage_index,
            status=resp.status,
            warnings=response_warnings,
        )


__all__ = ["CloudLauncher", "CloudLaunchResult"]
