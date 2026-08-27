from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

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
    CloudProvider,
    CloudRegion,
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
_CLOUD_PROVIDER_KEYS = {"modal": "modal", "aws": "aws_batch"}
_SUPPORTED_CLOUDS = frozenset({"aws", "oci", "gcp"})
_SUPPORTED_REGIONS = frozenset(
    {
        "us",
        "eu",
        "ca",
        "uk",
        "us-east",
        "us-central",
        "us-south",
        "us-west",
        "eu-west",
        "eu-north",
        "eu-south",
    }
)
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


def _normalize_cloud(value: str) -> CloudProvider:
    if value not in _SUPPORTED_CLOUDS:
        supported = ", ".join(sorted(_SUPPORTED_CLOUDS))
        raise ValueError(f"cloud must be one of: {supported}")
    return cast(CloudProvider, value)


def _normalize_regions(value: str | Sequence[str]) -> tuple[CloudRegion, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError("region must contain at least one selector")
    invalid = sorted({item for item in values if item not in _SUPPORTED_REGIONS})
    if invalid:
        supported = ", ".join(sorted(_SUPPORTED_REGIONS))
        raise ValueError(
            f"unsupported region selector(s): {', '.join(invalid)}; "
            f"expected one or more of: {supported}"
        )
    return cast(tuple[CloudRegion, ...], tuple(dict.fromkeys(values)))


@dataclass(frozen=True, slots=True)
class CloudLaunchResult:
    job_id: str
    stage_index: int
    status: str
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class PreparedDebugSync:
    pipeline_payload: bytes
    pipeline_sha256: str
    allocation_fingerprint: str


class CloudLauncher(BaseLauncher):
    """Cloud launcher that submits a compiled run to the cloud controller.

    Args:
        pipeline: Pipeline to execute.
        name: Human-readable run name.
        provider: Cloud compute provider. Supported values are ``"modal"`` and
            ``"aws"``.
        num_workers: Requested logical worker count for cloud execution, or
            ``"auto"`` to launch one worker per stage shard.
        cpus_per_worker: Optional requested CPU cores per worker.
        mem_mb_per_worker: Optional requested memory in MB per worker for cloud scheduling.
        gpu: Optional GPU runtime request for cloud scheduling.
        sync_local_dependencies: Whether to include packages detected from the
            local environment in the cloud runtime.
        dependencies: Additional packages to install in the cloud runtime.
            Entries are requirement strings.
        refiner_extras: Additional macrodata-refiner extras to install in the
            cloud runtime. Built-in blocks automatically declare the extras they
            require; pass this for extras used outside those blocks.
        secrets: Optional secret sources mounted into the cloud runtime.
        env: Optional plain environment variables mounted into the cloud runtime.
    """

    def __init__(
        self,
        *,
        pipeline: "RefinerPipeline",
        name: str,
        provider: str = "modal",
        num_workers: int | Literal["auto"] = 1,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpu: GPU | None = None,
        cloud: CloudProvider = "aws",
        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),
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
        normalized_provider = provider.strip().lower()
        if normalized_provider not in _CLOUD_PROVIDER_KEYS:
            raise ValueError("provider must be 'modal' or 'aws'")
        if normalized_provider == "aws" and gpu is not None:
            raise ValueError("provider='aws' does not support GPU workers")
        self.provider = normalized_provider
        self.cpus_per_worker = cpus_per_worker
        self.mem_mb_per_worker = mem_mb_per_worker
        self.cloud = _normalize_cloud(cloud)
        self.region = _normalize_regions(region)
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
        self, *, secret_values: tuple[str, ...], stages: list[PlannedStage]
    ) -> dict[str, object]:
        manifest = build_run_manifest(
            secret_values=secret_values,
            capture_dependencies=self.sync_local_dependencies,
            dependencies=self.dependencies,
            refiner_extras=self.refiner_extras,
            pipeline_stages=stages,
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

    def _resolve_submission(
        self,
    ) -> tuple[
        list[PlannedStage],
        dict[str, object],
        dict[str, object],
        list[dict[str, Any]] | None,
        dict[str, str] | None,
    ]:
        resolved_secret_sources, secret_values = resolve_secret_sources(self.secrets)
        resolved_env = resolve_env_mapping(self.env) if self.env else None
        stages = self._resolved_stages()
        manifest = self._resolve_cloud_manifest(
            secret_values=secret_values,
            stages=stages,
        )
        plan = self._compiled_plan(stages, secret_values=secret_values)
        return stages, manifest, plan, resolved_secret_sources, resolved_env

    def _secret_mount_spec(
        self,
        resolved_secret_sources: list[dict[str, Any]] | None,
        *,
        workspace_secret_versions: dict[str, list[dict[str, str]]] | None = None,
    ) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for source in resolved_secret_sources or []:
            if source.get("__type__") != "__envkeys__":
                specs.append(
                    {
                        "kind": "dict",
                        "value_digests": {
                            key: hashlib.sha256(str(value).encode()).hexdigest()
                            for key, value in sorted(source.items())
                        },
                    }
                )
            else:
                env_name = str(source.get("envname") or "default")
                selected_keys = (
                    set(source["keys"])
                    if isinstance(source.get("keys"), list)
                    else None
                )
                specs.append(
                    {
                        "kind": "env",
                        "name": env_name,
                        "keys": sorted(source["keys"])
                        if isinstance(source.get("keys"), list)
                        else None,
                        "versions": [
                            secret
                            for secret in (workspace_secret_versions or {}).get(
                                env_name, []
                            )
                            if selected_keys is None or secret["name"] in selected_keys
                        ],
                    }
                )
        return specs

    @staticmethod
    def _workspace_secret_versions(
        *,
        client: MacrodataClient | None,
        resolved_secret_sources: list[dict[str, Any]] | None,
    ) -> dict[str, list[dict[str, str]]]:
        env_names = {
            str(source.get("envname") or "default")
            for source in resolved_secret_sources or []
            if source.get("__type__") == "__envkeys__"
        }
        versions: dict[str, list[dict[str, str]]] = {}
        if not env_names:
            return versions
        resolved_client = client or MacrodataClient()
        for env_name in sorted(env_names):
            payload = resolved_client.cli_list_secrets(env=env_name)
            secrets = payload.get("secrets")
            if not isinstance(secrets, list):
                raise ValueError("invalid workspace secret metadata response")
            env_versions: list[dict[str, str]] = []
            for secret in secrets:
                if not isinstance(secret, dict):
                    raise ValueError("invalid workspace secret metadata response")
                secret_id = secret.get("id")
                name = secret.get("name")
                version = secret.get("version")
                if not all(
                    isinstance(value, str) and value for value in (secret_id, name)
                ) or not (
                    isinstance(version, str)
                    and len(version) == 64
                    and all(character in "0123456789abcdef" for character in version)
                ):
                    raise ValueError("invalid workspace secret metadata response")
                env_versions.append({"id": secret_id, "name": name, "version": version})
            versions[env_name] = sorted(
                env_versions, key=lambda secret: (secret["name"], secret["id"])
            )
        return versions

    def _debug_allocation_fingerprint(
        self,
        *,
        stages: list[PlannedStage],
        manifest: dict[str, object],
        resolved_secret_sources: list[dict[str, Any]] | None,
        resolved_env: dict[str, str] | None,
        workspace_secret_versions: dict[str, list[dict[str, str]]] | None = None,
    ) -> str:
        stage_specs: list[dict[str, object]] = []
        for stage in stages:
            runtime = CloudRuntimeConfig(
                num_workers=1,
                cloud=self.cloud,
                region=self.region,
                cpus_per_worker=stage.compute.cpus_per_worker,
                mem_mb_per_worker=stage.compute.memory_mb_per_worker,
                gpu=stage.compute.gpu,
            ).to_dict()
            runtime.pop("num_workers", None)
            stage_specs.append(
                {
                    "stage_index": stage.index,
                    "runtime": runtime,
                    "runtime_services": [
                        service.to_dict()
                        for service in collect_pipeline_services(stage.pipeline)
                    ],
                }
            )
        allocation = {
            "schema_version": 1,
            "provider": self.provider,
            "environment": manifest.get("environment"),
            "dependencies": manifest.get("dependencies"),
            "stages": stage_specs,
            "secret_mounts": self._secret_mount_spec(
                resolved_secret_sources,
                workspace_secret_versions=workspace_secret_versions,
            ),
            "env": resolved_env,
        }
        encoded = json.dumps(
            allocation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def prepare_debug_sync(
        self, *, client: MacrodataClient | None = None
    ) -> PreparedDebugSync:
        if self.continue_from_job is not None:
            raise ValueError("cloud debug cannot be combined with continue_from_job")
        stages, manifest, _, resolved_secret_sources, resolved_env = (
            self._resolve_submission()
        )
        stage = next((item for item in stages if item.index == 0), None)
        if stage is None:
            raise ValueError("pipeline has no stage 0")
        serialized = PreparedPipelinePayload.from_pipeline(stage.pipeline)
        workspace_secret_versions = self._workspace_secret_versions(
            client=client,
            resolved_secret_sources=resolved_secret_sources,
        )
        return PreparedDebugSync(
            pipeline_payload=serialized.payload_bytes,
            pipeline_sha256=serialized.sha256,
            allocation_fingerprint=self._debug_allocation_fingerprint(
                stages=stages,
                manifest=manifest,
                resolved_secret_sources=resolved_secret_sources,
                resolved_env=resolved_env,
                workspace_secret_versions=workspace_secret_versions,
            ),
        )

    def launch(self) -> CloudLaunchResult:
        from refiner.launchers.cloud_debug_capture import active_cloud_launch_capture

        capture = active_cloud_launch_capture()
        if capture is not None:
            return capture.capture(self)
        return self._launch(debug=False)

    def launch_debug(self) -> CloudLaunchResult:
        return self._launch(debug=True)

    def _launch(self, *, debug: bool) -> CloudLaunchResult:
        if debug and self.continue_from_job is not None:
            raise ValueError("cloud debug cannot be combined with continue_from_job")
        try:
            client = MacrodataClient()
        except MacrodataCredentialsError as err:
            raise SystemExit(
                "Launching jobs in the Macrodata cloud requires Macrodata "
                "authentication. Run `macrodata login` or set MACRODATA_API_KEY."
            ) from err
        stages, manifest, plan, resolved_secret_sources, resolved_env = (
            self._resolve_submission()
        )
        if debug:
            workspace_secret_versions = self._workspace_secret_versions(
                client=client,
                resolved_secret_sources=resolved_secret_sources,
            )
            manifest = dict(manifest)
            manifest["debug_allocation_fingerprint"] = (
                self._debug_allocation_fingerprint(
                    stages=stages,
                    manifest=manifest,
                    resolved_secret_sources=resolved_secret_sources,
                    resolved_env=resolved_env,
                    workspace_secret_versions=workspace_secret_versions,
                )
            )
        try:
            if self.provider == "aws" and any(
                collect_pipeline_services(stage.pipeline) for stage in stages
            ):
                raise ValueError(
                    "provider='aws' does not support managed runtime services"
                )
            pipeline_payloads = self._upload_stage_payloads(
                client=client, stages=stages
            )
            request = CloudRunCreateRequest(
                name=self.name,
                plan=plan,
                provider=_CLOUD_PROVIDER_KEYS[self.provider],
                stage_payloads=[
                    StagePayload(
                        stage_index=stage.index,
                        pipeline_payload=pipeline_payloads[stage.index],
                        runtime=CloudRuntimeConfig(
                            num_workers=stage.compute.num_workers,
                            cloud=self.cloud,
                            region=self.region,
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
                debug=debug,
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
        print(f"Cloud job launched. View job:\n  {tracking_url}", flush=True)
        if debug:
            return CloudLaunchResult(
                job_id=resp.job_id,
                stage_index=resp.stage_index,
                status=resp.status,
                warnings=response_warnings,
            )
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


__all__ = ["CloudLauncher", "CloudLaunchResult", "PreparedDebugSync"]
