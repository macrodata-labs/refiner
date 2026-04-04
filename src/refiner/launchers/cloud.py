from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from refiner.cli.ui import stdin_is_interactive
from refiner.platform.client import (
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    StagePayload,
    serialize_pipeline_inline,
)
from refiner.platform.manifest import refiner_ref_exists_on_remote

from refiner.launchers.base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


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
        heartbeat_interval_seconds: Worker heartbeat cadence.
        cpus_per_worker: Optional requested CPU cores per worker.
        mem_mb_per_worker: Optional requested memory in MB per worker for cloud scheduling.
        gpus_per_worker: Optional requested GPU count per worker for cloud scheduling.
        gpu_type: Optional requested GPU type per worker for cloud scheduling.
    """

    def __init__(
        self,
        *,
        pipeline: "RefinerPipeline",
        name: str,
        num_workers: int = 1,
        heartbeat_interval_seconds: int = 30,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        gpus_per_worker: int | None = None,
        gpu_type: str | None = None,
        sync_local_dependencies: bool = True,
        secrets: dict[str, object | None] | None = None,
        env: dict[str, object | None] | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            cpus_per_worker=cpus_per_worker,
            gpus_per_worker=gpus_per_worker,
        )
        if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
            raise ValueError("mem_mb_per_worker must be > 0")
        if (gpus_per_worker is not None) != (gpu_type is not None):
            raise ValueError("gpus_per_worker and gpu_type must be specified together or not at all")
        if gpu_type is not None and not gpu_type.strip():
            raise ValueError("gpu_type must be non-empty")
        self.sync_local_dependencies = sync_local_dependencies
        self.mem_mb_per_worker = mem_mb_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.gpu_type = gpu_type.strip() if gpu_type is not None else None
        self.secrets = secrets
        self.env = env

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
            client = self._require_platform_client()
        except RuntimeError as err:
            raise SystemExit(str(err)) from err
        resolved_secrets = self._resolve_env_values(self.secrets)
        resolved_env = self._resolve_env_values(self.env)
        secret_values = tuple(resolved_secrets.values()) if resolved_secrets else ()
        stages = self._planned_stages()
        manifest = self._resolve_cloud_manifest(secret_values=secret_values)
        request = CloudRunCreateRequest(
            name=self.name,
            plan=self._compiled_plan(stages, secret_values=secret_values),
            stage_payloads=[
                StagePayload(
                    stage_index=stage.index,
                    pipeline_payload=serialize_pipeline_inline(stage.pipeline),
                    runtime=CloudRuntimeConfig(
                        num_workers=stage.compute.num_workers,
                        heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                        cpus_per_worker=self.cpus_per_worker,
                        mem_mb_per_worker=self.mem_mb_per_worker,
                        gpus_per_worker=self.gpus_per_worker,
                        gpu_type=self.gpu_type,
                    ),
                )
                for stage in stages
            ],
            manifest=manifest,
            sync_local_dependencies=self.sync_local_dependencies,
            secrets=self._merged_env(resolved_secrets, resolved_env),
        )
        resp = client.cloud_submit_job(request=request)
        self._info(
            f"Track job here: {self._job_tracking_url(client=client, job_id=resp.job_id, workspace_slug=resp.workspace_slug)}"
        )
        return CloudLaunchResult(
            job_id=resp.job_id,
            stage_index=resp.stage_index,
            status=resp.status,
        )

    def _stage_resource_hints(self, *, stage: object) -> dict[str, object]:
        del stage
        hints: dict[str, object] = {}
        if self.cpus_per_worker is not None:
            hints["cpus_per_worker"] = self.cpus_per_worker
        if self.mem_mb_per_worker is not None:
            hints["memory_mb_per_worker"] = self.mem_mb_per_worker
        if self.gpus_per_worker is not None:
            hints["gpus_per_worker"] = self.gpus_per_worker
        if self.gpu_type is not None:
            hints["gpu_type"] = self.gpu_type
        return hints


__all__ = ["CloudLauncher", "CloudLaunchResult"]
