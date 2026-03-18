from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from refiner.platform.client import (
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    StagePayload,
    serialize_pipeline_inline,
)

from refiner.launchers.base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


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
        )
        if mem_mb_per_worker is not None and mem_mb_per_worker <= 0:
            raise ValueError("mem_mb_per_worker must be > 0")
        self.sync_local_dependencies = sync_local_dependencies
        self.mem_mb_per_worker = mem_mb_per_worker
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

    def launch(self) -> CloudLaunchResult:
        try:
            client = self._require_platform_client()
        except RuntimeError as err:
            raise SystemExit(str(err)) from err
        resolved_secrets = self._resolve_env_values(self.secrets)
        resolved_env = self._resolve_env_values(self.env)
        secret_values = tuple(resolved_secrets.values()) if resolved_secrets else ()
        stages = self._planned_stages()
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
                    ),
                )
                for stage in stages
            ],
            manifest=self._run_manifest(secret_values=secret_values),
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


__all__ = ["CloudLauncher", "CloudLaunchResult"]
