from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from refiner.platform.client import CloudRunCreateRequest, CloudRuntimeConfig
from refiner.platform.client import serialize_pipeline_inline

from .base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class CloudLaunchResult:
    job_id: str
    stage_id: str
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

    def launch(self) -> CloudLaunchResult:
        try:
            client = self._require_platform_client()
        except RuntimeError as err:
            raise SystemExit(str(err)) from err
        request = CloudRunCreateRequest(
            name=self.name,
            plan=self._compiled_plan(),
            runtime=CloudRuntimeConfig(
                num_workers=self.num_workers,
                heartbeat_interval_seconds=self.heartbeat_interval_seconds,
                cpus_per_worker=self.cpus_per_worker,
                mem_mb_per_worker=self.mem_mb_per_worker,
            ),
            pipeline_payload=serialize_pipeline_inline(self.pipeline),
            manifest=self._run_manifest(),
            sync_local_dependencies=self.sync_local_dependencies,
        )
        resp = client.cloud_submit_job(request=request)
        self._info(
            f"Track job here: {self._job_tracking_url(client=client, job_id=resp.job_id)}"
        )
        return CloudLaunchResult(
            job_id=resp.job_id,
            stage_id=resp.stage_id,
            status=resp.status,
        )


__all__ = ["CloudLauncher", "CloudLaunchResult"]
