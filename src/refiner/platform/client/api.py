from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..auth import current_api_key
from .http import parse_json_response, request_json
from .models import (
    CloudRunCreateRequest,
    CloudRunCreateResponse,
    CreateJobEnvelope,
    CreateJobResponse,
    OkResponse,
    RunHandle,
    ShardClaimResponse,
    ShardDescriptor,
    VerifyApiKeyResponse,
    WorkerStartedResponse,
)

if TYPE_CHECKING:
    from refiner.pipeline.data.shard import Shard


PLATFORM_BASE_URL_ENV_VAR = "MACRODATA_BASE_URL"
_PLATFORM_BASE_URL = "https://macrodata.co"


def resolve_platform_base_url() -> str:
    env_value = os.environ.get(PLATFORM_BASE_URL_ENV_VAR)
    if env_value:
        return env_value.rstrip("/")
    return _PLATFORM_BASE_URL


def compile_shard_descriptors(shards: list["Shard"]) -> list[ShardDescriptor]:
    return [ShardDescriptor.from_shard(shard) for shard in shards]


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
            stage_id=created_job.stage_id,
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
        self, *, job_id: str, stage_id: str, shards: list["Shard"]
    ) -> OkResponse:
        shard_descriptors = compile_shard_descriptors(shards)
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/shards/register",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"shards": [shard.to_dict() for shard in shard_descriptors]},
        )
        return parse_json_response(response_data, OkResponse)

    def report_worker_started(
        self,
        *,
        job_id: str,
        stage_id: str,
        host: str | None = None,
        worker_name: str | None = None,
    ) -> WorkerStartedResponse:
        request_body: dict[str, Any] = {}
        if host:
            request_body["host"] = host
        if worker_name:
            request_body["name"] = worker_name

        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/start",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
        )
        return parse_json_response(response_data, WorkerStartedResponse)

    def report_worker_finished(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        status: str,
        error: str | None = None,
    ) -> OkResponse:
        request_body: dict[str, Any] = {"status": status}
        if error:
            request_body["error"] = error
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/workers/{worker_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
        )
        return parse_json_response(response_data, OkResponse)

    def report_stage_finished(
        self, *, job_id: str, stage_id: str, status: str
    ) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
        )
        return parse_json_response(response_data, OkResponse)

    def report_job_finished(self, *, job_id: str, status: str) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"status": status},
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
        stage_id: str,
        worker_id: str,
        previous_shard_id: str | None = None,
    ) -> ShardClaimResponse:
        request_body: dict[str, Any] = {"worker_id": worker_id}
        if previous_shard_id:
            request_body["previous_shard_id"] = previous_shard_id
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/shards/claim",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
        )
        return parse_json_response(response_data, ShardClaimResponse)

    def shard_heartbeat(
        self, *, job_id: str, stage_id: str, worker_id: str, shard_ids: list[str]
    ) -> OkResponse:
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/shards/heartbeat",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload={"worker_id": worker_id, "shard_ids": shard_ids},
        )
        return parse_json_response(response_data, OkResponse)

    def shard_finish(
        self,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        shard_id: str,
        status: str,
        error: str | None = None,
    ) -> OkResponse:
        request_body: dict[str, Any] = {"worker_id": worker_id, "status": status}
        if error:
            request_body["error"] = error
        response_data = request_json(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_id}/shards/{shard_id}/finish",
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=request_body,
        )
        return parse_json_response(response_data, OkResponse)


__all__ = [
    "MacrodataClient",
    "RunHandle",
    "compile_shard_descriptors",
    "resolve_platform_base_url",
    "verify_api_key",
]
