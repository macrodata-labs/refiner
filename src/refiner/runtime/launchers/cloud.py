from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from refiner.platform import MacrodataClient
from refiner.platform.client import compile_shard_descriptors
from refiner.platform.auth import current_api_key
from refiner.platform.cloud.models import CloudRunCreateRequest, CloudRuntimeConfig
from refiner.platform.cloud.serialize import serialize_pipeline_inline
from refiner.runtime.planning import compile_pipeline_plan

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
        heartbeat_every_rows: Worker heartbeat cadence.
        cpus_per_worker: Optional requested CPU cores per worker.
        mem_mb_per_worker: Optional requested memory in MB per worker.
    """

    def __init__(
        self,
        *,
        pipeline: "RefinerPipeline",
        name: str,
        num_workers: int = 1,
        heartbeat_every_rows: int = 4096,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_every_rows=heartbeat_every_rows,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )

    def launch(self) -> CloudLaunchResult:
        api_key = current_api_key()
        client = MacrodataClient(api_key=api_key)
        request = CloudRunCreateRequest(
            name=self.name,
            plan=compile_pipeline_plan(self.pipeline),
            runtime=CloudRuntimeConfig(
                num_workers=self.num_workers,
                heartbeat_every_rows=self.heartbeat_every_rows,
                cpus_per_worker=self.cpus_per_worker,
                mem_mb_per_worker=self.mem_mb_per_worker,
            ),
            pipeline_payload=serialize_pipeline_inline(self.pipeline),
            shards=compile_shard_descriptors(list(self.pipeline.source.list_shards())),
        )
        resp = client.cloud_submit_job(request=request)
        return CloudLaunchResult(
            job_id=resp.job_id,
            stage_id=resp.stage_id,
            status=resp.status,
        )


__all__ = ["CloudLauncher", "CloudLaunchResult"]
