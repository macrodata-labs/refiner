from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any

from .config import resolve_platform_base_url
from .http import MacrodataApiError, request_json
from .telemetry import OtelTelemetryEmitter, WorkerTelemetry
from refiner.runtime.planning import compile_pipeline_plan

if TYPE_CHECKING:
    from refiner.ledger.shard import Shard
    from refiner.pipeline import RefinerPipeline


def compile_shard_descriptors(shards: list["Shard"]) -> list[dict[str, Any]]:
    return [shard.to_dict() for shard in shards]


@dataclass(frozen=True, slots=True)
class ObserverJobContext:
    job_id: str
    stage_id: str


class ObserverClient:
    def __init__(self, *, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = (base_url or resolve_platform_base_url()).rstrip("/")

    def submit_job(
        self, *, name: str, pipeline: "RefinerPipeline"
    ) -> ObserverJobContext:
        payload = {
            "name": name,
            "executor": {"type": "refiner-local"},
            "plan": compile_pipeline_plan(pipeline),
        }
        resp = request_json(
            method="POST",
            path="/api/jobs",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )
        job = resp.get("job")
        if not isinstance(job, dict):
            raise MacrodataApiError(
                status=200, message="Missing job in /api/jobs response"
            )
        job_id = job.get("id")
        stages = job.get("stages")
        if not isinstance(job_id, str) or not job_id:
            raise MacrodataApiError(
                status=200, message="Missing job.id in /api/jobs response"
            )
        if not isinstance(stages, list) or not stages:
            raise MacrodataApiError(
                status=200, message="Missing stages in /api/jobs response"
            )
        stage0 = stages[0]
        if not isinstance(stage0, dict) or not isinstance(stage0.get("id"), str):
            raise MacrodataApiError(
                status=200, message="Missing stage id in /api/jobs response"
            )
        return ObserverJobContext(job_id=job_id, stage_id=stage0["id"])

    def register_stage_shards(
        self, *, job_id: str, stage_id: str, shards: list["Shard"]
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/shards/register",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shards": compile_shard_descriptors(shards)},
        )

    def start_worker(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        host: str | None = None,
        config: WorkerConfig | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "worker_id": worker_id,
        }
        if host is not None:
            payload["host"] = host
        if config is not None:
            payload["config"] = config.to_api_payload()
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )

    def start_shard(
        self, *, job_id: str, stage_id: str, worker_id: str, shard_id: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/shards/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shard_id": shard_id},
        )

    def finish_shard(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        shard_id: str,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"shard_id": shard_id, "status": status}
        if error:
            payload["error"] = error
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/shards/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )

    def finish_worker(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status}
        if error:
            payload["error"] = error
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )

    def finish_stage(
        self, *, job_id: str, stage_id: str, status: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
        )

    def finish_job(self, *, job_id: str, status: str) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
        )

    def worker_telemetry(
        self, *, job_id: str, stage_id: str, worker_id: str
    ) -> WorkerTelemetry:
        return OtelTelemetryEmitter(
            base_url=self.base_url,
            api_key=self.api_key,
            job_id=job_id,
            stage_id=stage_id,
            worker_id=worker_id,
        )


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Worker resource config for start-worker registration. Omit fields to leave unset."""

    cpu_cores: int | None = None
    memory_mb: int | None = None
    gpu_count: int | None = None

    @classmethod
    def from_runtime(cls) -> "WorkerConfig":
        """Build a best-effort worker config from the current runtime."""
        cpu_cores = os.cpu_count()
        memory_mb: int | None = None
        try:
            import psutil

            memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        except Exception:
            memory_mb = None
        return cls(cpu_cores=cpu_cores, memory_mb=memory_mb, gpu_count=None)

    def to_api_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.cpu_cores is not None:
            out["cpuCores"] = self.cpu_cores
        if self.memory_mb is not None:
            out["memoryMb"] = self.memory_mb
        if self.gpu_count is not None:
            out["gpuCount"] = self.gpu_count
        return out


__all__ = ["ObserverClient", "ObserverJobContext", "WorkerConfig"]
