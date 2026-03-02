from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refiner.runtime.planning import compile_pipeline_plan

from .cloud.models import CloudRunCreateRequest, CloudRunCreateResponse
from .config import resolve_platform_base_url
from .http import MacrodataApiError, request_json

if TYPE_CHECKING:
    from refiner.ledger.shard import Shard
    from refiner.pipeline import RefinerPipeline


def compile_shard_descriptors(shards: list["Shard"]) -> list[dict[str, Any]]:
    return [shard.to_dict() for shard in shards]


@dataclass(frozen=True, slots=True)
class JobContext:
    job_id: str
    stage_id: str


class MacrodataClient:
    def __init__(self, *, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = (base_url or resolve_platform_base_url()).rstrip("/")

    def create_job(self, *, name: str, pipeline: "RefinerPipeline") -> JobContext:
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
        if not isinstance(stage0, dict):
            raise MacrodataApiError(
                status=200, message="Missing stage id in /api/jobs response"
            )
        stage_id = stage0.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            raise MacrodataApiError(
                status=200, message="Missing stage id in /api/jobs response"
            )
        return JobContext(job_id=job_id, stage_id=stage_id)

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

    def report_worker_started(
        self, *, job_id: str, stage_id: str, worker_id: str, host: str | None = None
    ) -> dict[str, Any]:
        payload = {"host": host} if host else {}
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )

    def report_shard_started(
        self, *, job_id: str, stage_id: str, worker_id: str, shard_id: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/shards/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shard_id": shard_id},
        )

    def report_shard_finished(
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

    def report_worker_finished(
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

    def report_stage_finished(
        self, *, job_id: str, stage_id: str, status: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
        )

    def report_job_finished(self, *, job_id: str, status: str) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
        )

    def cloud_submit_job(
        self, *, request: CloudRunCreateRequest
    ) -> CloudRunCreateResponse:
        payload = request_json(
            method="POST",
            path="/api/cloud/runs",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request.to_dict(),
            timeout_s=60.0,
        )

        def required_str(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                raise MacrodataApiError(
                    status=200,
                    message=f"Missing {key} in /api/cloud/runs response",
                )
            return value

        return CloudRunCreateResponse(
            job_id=required_str("job_id"),
            stage_id=required_str("stage_id"),
            status=required_str("status"),
        )

    def cloud_ledger_register_stage_shards(
        self, *, job_id: str, stage_id: str, shards: list["Shard"]
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/cloud/ledger/jobs/{job_id}/stages/{stage_id}/shards/register",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shards": compile_shard_descriptors(shards)},
        )

    def cloud_ledger_claim_shard(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        previous_shard_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"worker_id": worker_id}
        if previous_shard_id:
            payload["previous_shard_id"] = previous_shard_id
        return request_json(
            method="POST",
            path=f"/api/cloud/ledger/jobs/{job_id}/stages/{stage_id}/shards/claim",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )

    def cloud_ledger_heartbeat_shard(
        self, *, job_id: str, stage_id: str, worker_id: str, shard_id: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/cloud/ledger/jobs/{job_id}/stages/{stage_id}/shards/heartbeat",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"worker_id": worker_id, "shard_id": shard_id},
        )

    def cloud_ledger_complete_shard(
        self, *, job_id: str, stage_id: str, worker_id: str, shard_id: str
    ) -> dict[str, Any]:
        return request_json(
            method="POST",
            path=f"/api/cloud/ledger/jobs/{job_id}/stages/{stage_id}/shards/complete",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"worker_id": worker_id, "shard_id": shard_id},
        )

    def cloud_ledger_fail_shard(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        shard_id: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"worker_id": worker_id, "shard_id": shard_id}
        if error:
            payload["error"] = error
        return request_json(
            method="POST",
            path=f"/api/cloud/ledger/jobs/{job_id}/stages/{stage_id}/shards/fail",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=payload,
        )


__all__ = ["MacrodataClient", "JobContext", "compile_shard_descriptors"]
