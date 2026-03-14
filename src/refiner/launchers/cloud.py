from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from refiner.platform.client import (
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    StagePayload,
)
from refiner.platform.client import serialize_pipeline_inline
from refiner.platform.manifest import _redact_captured_strings

from .base import BaseLauncher

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
        mem_mb_per_worker: Optional requested memory in MB per worker.
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
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )
        self.sync_local_dependencies = sync_local_dependencies
        self.secrets = secrets

    def _resolved_secrets(self) -> dict[str, str] | None:
        if not self.secrets:
            return None
        resolved: dict[str, str] = {}
        for name, value in self.secrets.items():
            if value is None:
                env_value = os.environ.get(name)
                if env_value is None:
                    raise SystemExit(
                        f"cloud secret {name!r} was set to None but is not present in the environment. Make sure it is being exported."
                    )
                resolved[name] = env_value
                continue
            resolved[name] = str(value)
        return resolved

    def _redact_request(
        self,
        request: CloudRunCreateRequest,
        *,
        secret_values: tuple[str, ...],
    ) -> CloudRunCreateRequest:
        if not secret_values:
            return request
        return CloudRunCreateRequest(
            name=request.name,
            plan=_redact_captured_strings(request.plan, secret_values=secret_values),
            stage_payloads=request.stage_payloads,
            manifest=(
                _redact_captured_strings(request.manifest, secret_values=secret_values)
                if request.manifest is not None
                else None
            ),
            sync_local_dependencies=request.sync_local_dependencies,
            secrets=request.secrets,
        )

    def launch(self) -> CloudLaunchResult:
        try:
            client = self._require_platform_client()
        except RuntimeError as err:
            raise SystemExit(str(err)) from err
        resolved_secrets = self._resolved_secrets()
        secret_values = tuple(resolved_secrets.values()) if resolved_secrets else ()
        stages = self._planned_stages()
        request = CloudRunCreateRequest(
            name=self.name,
            plan=self._compiled_plan(stages),
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
            manifest=self._run_manifest(),
            sync_local_dependencies=self.sync_local_dependencies,
            secrets=resolved_secrets,
        )
        request = self._redact_request(request, secret_values=secret_values)
        resp = client.cloud_submit_job(request=request)
        self._info(
            f"Track job here: {self._job_tracking_url(client=client, job_id=resp.job_id)}"
        )
        return CloudLaunchResult(
            job_id=resp.job_id,
            stage_index=resp.stage_index,
            status=resp.status,
        )


__all__ = ["CloudLauncher", "CloudLaunchResult"]
