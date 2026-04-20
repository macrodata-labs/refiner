from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from refiner.cli.run.modes import (
    CloudAttachContext,
    emit_cloud_followup_commands,
    resolve_launcher_attach_mode,
)
from refiner.cli.ui import stdin_is_interactive, stdout_is_interactive
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudRunCreateRequest,
    CloudRunResumeRequest,
    CloudRuntimeConfig,
    CloudRuntimeOverrides,
    MacrodataApiError,
    MacrodataClient,
    StagePayload,
    serialize_pipeline_inline,
)
from refiner.platform.client.models import build_resume_selector
from refiner.platform.manifest import refiner_ref_exists_on_remote

from refiner.job_urls import build_job_tracking_url
from refiner.launchers.base import BaseLauncher
from refiner.pipeline.planning import plan_pipeline_stages

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline
    from refiner.pipeline.planning import PlannedStage


_FALLBACK_ENV_VAR = "MACRODATA_FALLBACK_TO_LATEST_PYPI"


@dataclass(frozen=True, slots=True)
class CloudLaunchResult:
    job_id: str
    stage_index: int
    status: str


class CloudLauncher(BaseLauncher):
    """Cloud launcher that submits a compiled run to the cloud controller.

    Args:
        pipeline: Pipeline to execute.
        name: Human-readable run name.
        num_workers: Requested logical worker count for cloud execution.
        cpus_per_worker: Optional requested CPU cores per worker.
        mem_mb_per_worker: Optional requested memory in MB per worker for cloud scheduling.
        gpus_per_worker: Optional requested GPU count per worker for cloud scheduling.
        gpu_type: Optional requested GPU type per worker for cloud scheduling.
        sync_local_dependencies: Whether to sync submitting environment dependencies.
        secrets: Optional secrets mounted into the cloud runtime.
        env: Optional plain environment variables mounted into the cloud runtime.
    """

    def __init__(
        self,
        *,
        pipeline: "RefinerPipeline",
        name: str,
        num_workers: int | None = None,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpus_per_worker: int | None = None,
        gpu_type: str | None = None,
        sync_local_dependencies: bool = True,
        secrets: dict[str, object | None] | None = None,
        env: dict[str, object | None] | None = None,
        resume_from_job_id: str | None = None,
        resume: str | None = None,
        resume_name: str | None = None,
        resume_limit_to_me: bool = False,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=1 if num_workers is None else num_workers,
            cpus_per_worker=cpus_per_worker,
            gpus_per_worker=gpus_per_worker,
        )
        if resume is not None and resume != "latest-compatible":
            raise ValueError("resume must be 'latest-compatible' when provided")
        resume_selector = build_resume_selector(
            job_id=resume_from_job_id,
            latest_compatible=(resume == "latest-compatible"),
            name=resume_name,
            limit_to_me=resume_limit_to_me,
        )
        if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
            raise ValueError("mem_mb_per_worker must be > 0")
        normalized_gpu_type = gpu_type.strip() if gpu_type is not None else None
        if gpu_type is not None and not normalized_gpu_type:
            raise ValueError("gpu_type must be non-empty")
        if resume_selector is None:
            if gpus_per_worker is not None and normalized_gpu_type is None:
                raise ValueError("gpu_type is required when gpus_per_worker is set")
            if normalized_gpu_type is not None and gpus_per_worker is None:
                raise ValueError("gpus_per_worker is required when gpu_type is set")
        self.cpus_per_worker = cpus_per_worker
        self.mem_mb_per_worker = mem_mb_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.gpu_type = normalized_gpu_type
        self.runtime_override_num_workers = num_workers
        self.sync_local_dependencies = sync_local_dependencies
        self.secrets = secrets
        self.env = env
        self.resume_selector = resume_selector

    @staticmethod
    def _resolve_env_values(
        values: dict[str, object | None] | None,
    ) -> dict[str, str] | None:
        if not values:
            return None
        resolved: dict[str, str] = {}
        for name, value in values.items():
            if value is None:
                env_value = os.environ.get(name)
                if env_value is None:
                    raise SystemExit(
                        f"cloud env {name!r} was set to None but is not present in the environment. Make sure it is being exported."
                    )
                resolved[name] = env_value
                continue
            resolved[name] = str(value)
        return resolved

    @staticmethod
    def _merged_env(
        secrets: dict[str, str] | None,
        env: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if secrets and env:
            overlapping = secrets.keys() & env.keys()
            if overlapping:
                raise SystemExit(
                    "cloud env keys must not overlap with secrets: "
                    + ", ".join(sorted(overlapping))
                )
        return {**(secrets or {}), **(env or {})} or None

    @staticmethod
    def _fallback_to_latest_pypi_enabled() -> bool:
        raw = os.environ.get(_FALLBACK_ENV_VAR, "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_cloud_manifest(
        self, *, secret_values: tuple[str, ...]
    ) -> dict[str, object]:
        manifest = self._run_manifest(secret_values=secret_values)
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

    def launch(self) -> CloudLaunchResult:
        try:
            client = MacrodataClient()
        except MacrodataCredentialsError as err:
            raise SystemExit(
                "Launching jobs in the Macrodata cloud requires Macrodata "
                "authentication. Run `macrodata login` or set MACRODATA_API_KEY."
            ) from err
        resolved_secrets = self._resolve_env_values(self.secrets)
        resolved_env = self._resolve_env_values(self.env)
        secret_values = tuple(resolved_secrets.values()) if resolved_secrets else ()
        selector = self.resume_selector
        merged_env = self._merged_env(resolved_secrets, resolved_env)
        try:
            if selector is None:
                manifest = self._resolve_cloud_manifest(secret_values=secret_values)
                planned_stages = self._planned_stages()
                stages = self._resolved_stages(planned_stages)
                compiled_plan = self._compiled_plan(
                    planned_stages, secret_values=secret_values
                )
                stage_payloads = [
                    StagePayload(
                        stage_index=stage.index,
                        pipeline_payload=serialize_pipeline_inline(stage.pipeline),
                        runtime=CloudRuntimeConfig(
                            num_workers=stage.compute.num_workers,
                            cpus_per_worker=stage.compute.cpus_per_worker,
                            mem_mb_per_worker=stage.compute.memory_mb_per_worker,
                            gpus_per_worker=stage.compute.gpus_per_worker,
                            gpu_type=stage.compute.gpu_type,
                        ),
                    )
                    for stage in stages
                ]
                resp = client.cloud_submit_job(
                    request=CloudRunCreateRequest(
                        name=self.name,
                        plan=compiled_plan,
                        stage_payloads=stage_payloads,
                        manifest=manifest,
                        sync_local_dependencies=self.sync_local_dependencies,
                        secrets=merged_env,
                    )
                )
            else:
                # Resume launches a new cloud job from the current compiled pipeline while
                # reusing completed work from a compatible failed job. The compiled stage
                # graph and manifest therefore always come from the current launch request.
                # Effective worker sizing for the resumed suffix is resolved by the backend
                # from the prior job plus explicit runtime_overrides, so resume planning uses
                # a placeholder worker count and strips runtime sizing from the compatibility
                # plan.
                manifest = self._resolve_cloud_manifest(secret_values=secret_values)
                stages = self._resolved_stages(
                    plan_pipeline_stages(self.pipeline, default_num_workers=1)
                )
                resp = client.cloud_resume_job(
                    request=CloudRunResumeRequest(
                        selector=selector,
                        name=self.name,
                        runtime_overrides=self._runtime_overrides(),
                        plan=self._resume_plan(stages, secret_values=secret_values),
                        stage_payloads=[
                            StagePayload(
                                stage_index=stage.index,
                                pipeline_payload=serialize_pipeline_inline(
                                    stage.pipeline
                                ),
                            )
                            for stage in stages
                        ],
                        manifest=manifest,
                        sync_local_dependencies=self.sync_local_dependencies,
                        secrets=merged_env,
                    )
                )
        except MacrodataCredentialsError as err:
            raise SystemExit(
                "Your Macrodata API key is invalid. Run `macrodata login` "
                "or set MACRODATA_API_KEY with a valid key."
            ) from err
        tracking_url = build_job_tracking_url(
            client=client,
            job_id=resp.job_id,
            workspace_slug=resp.workspace_slug,
        )
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
                if attach_rc != 0:
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
        )

    def _runtime_overrides(self) -> CloudRuntimeOverrides | None:
        if (
            self.runtime_override_num_workers is None
            and self.cpus_per_worker is None
            and self.mem_mb_per_worker is None
            and self.gpus_per_worker is None
            and self.gpu_type is None
        ):
            return None
        return CloudRuntimeOverrides(
            num_workers=self.runtime_override_num_workers,
            cpus_per_worker=self.cpus_per_worker,
            mem_mb_per_worker=self.mem_mb_per_worker,
            gpus_per_worker=self.gpus_per_worker,
            gpu_type=self.gpu_type,
        )

    def _resume_plan(
        self,
        stages: list["PlannedStage"],
        *,
        secret_values: tuple[str, ...],
    ) -> dict[str, object]:
        plan = self._compiled_plan(stages, secret_values=secret_values)
        resume_plan = dict(plan)
        stage_dicts = cast(list[dict[str, object]], resume_plan.get("stages", []))
        resume_plan["stages"] = [
            (
                {
                    key: value
                    for key, value in stage_dict.items()
                    if key
                    not in {
                        "requested_num_workers",
                        "cpus_per_worker",
                        "memory_mb_per_worker",
                        "gpus_per_worker",
                        "gpu_type",
                    }
                }
                if planned_stage.compute.inherit_launcher_resources
                else stage_dict
            )
            for planned_stage, stage_dict in zip(stages, stage_dicts, strict=True)
        ]
        return resume_plan


__all__ = ["CloudLauncher", "CloudLaunchResult"]
