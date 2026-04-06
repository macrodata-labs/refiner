from __future__ import annotations

from typing import TYPE_CHECKING, Any

from refiner.platform.auth import current_api_key
from refiner.platform.client.http import (
    parse_json_response,
    request_json,
    resolve_platform_base_url,
)
from refiner.platform.client.models import (
    CloudRunCreateRequest,
    CloudRunCreateResponse,
    CreateJobEnvelope,
    CreateJobResponse,
    FinalizedShardWorkersResponse,
    OkResponse,
    ShardClaimResponse,
    SerializedShard,
    VerifyApiKeyResponse,
    WorkerStartedResponse,
)
from refiner.worker.context import RunHandle

if TYPE_CHECKING:
    from refiner.pipeline.data.shard import Shard


def compile_shard_descriptors(shards: list["Shard"]) -> list[SerializedShard]:
    return [SerializedShard.from_shard(shard) for shard in shards]


def verify_api_key(
    *, api_key: str, base_url: str | None = None, timeout_s: float = 10.0
) -> VerifyApiKeyResponse:
    return MacrodataClient(api_key=api_key, base_url=base_url).verify_api_key(
        timeout_s=timeout_s
    )


class MacrodataClient:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key if api_key is not None else current_api_key()
        self.base_url = (base_url or resolve_platform_base_url()).rstrip("/")

    def create_job(
        self,
        *,
        name: str,
        executor: dict[str, Any],
        plan: dict[str, Any],
        manifest: dict[str, Any],
    ) -> RunHandle:
        request_body = {
            "name": name,
            "executor": executor,
            "plan": plan,
            "manifest": manifest,
        }
        response_data = request_json(
            method="POST",
            path="/api/jobs/submit",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
        )
        job_envelope = parse_json_response(response_data, CreateJobEnvelope)
        created_job = CreateJobResponse.from_envelope(job_envelope)
        return RunHandle(
            job_id=created_job.job_id,
            stage_index=created_job.stage_index,
            client=self,
            workspace_slug=created_job.workspace_slug,
        )

    def verify_api_key(self, *, timeout_s: float = 10.0) -> VerifyApiKeyResponse:
        response_data = request_json(
            method="GET",
            path="/api/me",
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=timeout_s,
        )
        return parse_json_response(response_data, VerifyApiKeyResponse)

    def shard_register(
        self, *, job_id: str, stage_index: int, shards: list["Shard"]
    ) -> OkResponse:
        shard_descriptors = compile_shard_descriptors(shards)
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/shards/register",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shards": [shard.to_dict() for shard in shard_descriptors]},
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def report_worker_started(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str | None = None,
        parent_provider_call_id: str | None = None,
        host: str | None = None,
        worker_name: str | None = None,
    ) -> WorkerStartedResponse:
        request_body: dict[str, Any] = {}
        if worker_id:
            request_body["worker_id"] = worker_id
        if parent_provider_call_id:
            request_body["parent_provider_call_id"] = parent_provider_call_id
        if host:
            request_body["host"] = host
        if worker_name:
            request_body["name"] = worker_name

        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
            timeout_s=60.0,
        )
        return parse_json_response(response_data, WorkerStartedResponse)

    def start_worker_services(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        services: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/{worker_id}/services/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"services": services},
            timeout_s=60.0,
        )
        if not isinstance(response_data, dict):
            raise ValueError("runtime services response must be a JSON object")
        return response_data

    def get_worker_services(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
    ) -> dict[str, Any]:
        response_data = request_json(
            method="GET",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/{worker_id}/services",
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=60.0,
        )
        if not isinstance(response_data, dict):
            raise ValueError("runtime services response must be a JSON object")
        return response_data

    def stop_worker_services(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
    ) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/{worker_id}/services/stop",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={},
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def report_worker_finished(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        status: str,
        error: str | None = None,
    ) -> OkResponse:
        request_body: dict[str, Any] = {"status": status}
        if status == "failed":
            request_body["error"] = (error or "").strip() or "UnknownError"
        elif error:
            request_body["error"] = error
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/{worker_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def report_stage_finished(
        self, *, job_id: str, stage_index: int, status: str
    ) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def report_job_finished(self, *, job_id: str, status: str) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def cloud_submit_job(
        self, *, request: CloudRunCreateRequest
    ) -> CloudRunCreateResponse:
        response_data = request_json(
            method="POST",
            path="/api/cloud/runs",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request.to_dict(),
            timeout_s=60.0,
        )
        return parse_json_response(response_data, CloudRunCreateResponse)

    def shard_claim(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        previous_shard_id: str | None = None,
    ) -> ShardClaimResponse:
        request_body: dict[str, Any] = {"worker_id": worker_id}
        if previous_shard_id:
            request_body["previous_shard_id"] = previous_shard_id
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/shards/claim",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
            timeout_s=60.0,
        )
        return parse_json_response(response_data, ShardClaimResponse)

    def shard_heartbeat(
        self, *, job_id: str, stage_index: int, worker_id: str, shard_ids: list[str]
    ) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/shards/heartbeat",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"worker_id": worker_id, "shard_ids": shard_ids},
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)

    def shard_finalized_workers(
        self, *, job_id: str, stage_index: int
    ) -> FinalizedShardWorkersResponse:
        response_data = request_json(
            method="GET",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/shards/finalized-workers",
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=60.0,
        )
        return parse_json_response(response_data, FinalizedShardWorkersResponse)

    def shard_finish(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        shard_id: str,
        status: str,
        error: str | None = None,
    ) -> OkResponse:
        request_body: dict[str, Any] = {"worker_id": worker_id, "status": status}
        if status == "failed":
            request_body["error"] = (error or "").strip() or "UnknownError"
        elif error:
            request_body["error"] = error
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/shards/{shard_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
            timeout_s=60.0,
        )
        return parse_json_response(response_data, OkResponse)


__all__ = [
    "MacrodataClient",
    "RunHandle",
    "compile_shard_descriptors",
    "resolve_platform_base_url",
    "verify_api_key",
]
