from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import msgspec

from refiner.platform.auth import MacrodataCredentialsError, current_api_key
from refiner.platform.client.models import (
    CloudRunCreateRequest,
    CloudRunCreateResponse,
    CreateJobEnvelope,
    CreateJobResponse,
    StageLifecycleResponse,
    VerifyApiKeyResponse,
)

T = TypeVar("T")
PLATFORM_BASE_URL_ENV_VAR = "MACRODATA_BASE_URL"
_PLATFORM_BASE_URL = "https://macrodata.co"

try:
    _REFINER_VERSION = importlib.metadata.version("macrodata-refiner").strip()
except importlib.metadata.PackageNotFoundError:
    _REFINER_VERSION = ""

_USER_AGENT = (
    f"macrodata-refiner/{_REFINER_VERSION}" if _REFINER_VERSION else "macrodata-refiner"
)


def sanitize_terminal_text(value: str) -> str:
    return "".join(ch for ch in value if ch >= " " and ch != "\x7f")


def resolve_platform_base_url() -> str:
    env_value = os.environ.get(PLATFORM_BASE_URL_ENV_VAR)
    if env_value:
        return env_value.rstrip("/")
    return _PLATFORM_BASE_URL


@dataclass
class MacrodataApiError(Exception):
    status: int
    message: str

    def __str__(self) -> str:
        return f"HTTP {self.status}: {self.message}"


def _decode_json_object(resp: httpx.Response, *, context: str) -> dict[str, Any]:
    try:
        payload = resp.json()
    except ValueError as err:
        raise MacrodataApiError(
            status=resp.status_code, message=f"Invalid JSON from {context}"
        ) from err
    if not isinstance(payload, dict):
        raise MacrodataApiError(
            status=resp.status_code, message=f"Unexpected response from {context}"
        )
    return payload


def _http_error_message(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        content_type = resp.headers.get("content-type", "").lower()
        text = resp.text.strip()
        if "html" in content_type:
            return sanitize_terminal_text(resp.reason_phrase or "HTTP error")
        if not text:
            return sanitize_terminal_text(resp.reason_phrase or "HTTP error")
        one_line = " ".join(text.split())
        if len(one_line) > 200:
            one_line = f"{one_line[:197]}..."
        return sanitize_terminal_text(one_line)
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("message")
        if isinstance(message, str) and message:
            return sanitize_terminal_text(message)
    return sanitize_terminal_text(resp.reason_phrase or "HTTP error")


def request_json(
    *,
    method: str,
    path: str,
    api_key: str | None = None,
    base_url: str | None = None,
    json_payload: dict[str, Any] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    resolved_base_url = resolve_platform_base_url() if base_url is None else base_url
    url = f"{resolved_base_url.rstrip('/')}{path}"
    headers = {"User-Agent": _USER_AGENT}
    if api_key is not None and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.request(
            method,
            url,
            headers=headers,
            json=json_payload,
            timeout=timeout_s,
        )
    except httpx.RequestError as err:
        raise MacrodataApiError(status=0, message=str(err)) from err

    if resp.is_error:
        if resp.status_code == 401:
            raise MacrodataCredentialsError(
                _http_error_message(resp),
                missing=False,
            )
        raise MacrodataApiError(
            status=resp.status_code, message=_http_error_message(resp)
        )

    return _decode_json_object(resp, context=path)


def verify_api_key(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout_s: float = 10.0,
) -> VerifyApiKeyResponse:
    return MacrodataClient(api_key=api_key, base_url=base_url).verify_api_key(
        timeout_s=timeout_s
    )


class MacrodataClient:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key if api_key is not None else current_api_key()
        self.base_url = (base_url or resolve_platform_base_url()).rstrip("/")

    def _request(
        self,
        *,
        method: str,
        path: str,
        response_type: type[T],
        json_payload: dict[str, Any] | None = None,
        timeout_s: float = 10.0,
    ) -> T:
        response_data = request_json(
            method=method,
            path=path,
            api_key=self.api_key,
            base_url=self.base_url,
            json_payload=json_payload,
            timeout_s=timeout_s,
        )
        try:
            return msgspec.convert(response_data, type=response_type, strict=True)
        except (TypeError, msgspec.ValidationError) as err:
            raise MacrodataApiError(status=200, message=str(err)) from err

    def create_job(
        self,
        *,
        name: str,
        executor: dict[str, Any],
        plan: dict[str, Any],
        manifest: dict[str, Any],
    ) -> CreateJobResponse:
        request_body = {
            "name": name,
            "executor": executor,
            "plan": plan,
            "manifest": manifest,
        }
        job_envelope = self._request(
            method="POST",
            path="/api/jobs/submit",
            response_type=CreateJobEnvelope,
            json_payload=request_body,
            timeout_s=30.0,
        )
        return CreateJobResponse.from_envelope(job_envelope)

    def verify_api_key(self, *, timeout_s: float = 10.0) -> VerifyApiKeyResponse:
        return self._request(
            method="GET",
            path="/api/me",
            response_type=VerifyApiKeyResponse,
            timeout_s=timeout_s,
        )

    def report_stage_started(
        self,
        *,
        job_id: str,
        stage_index: int,
    ) -> StageLifecycleResponse:
        return self._request(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/start",
            response_type=StageLifecycleResponse,
        )

    def report_stage_finished(
        self, *, job_id: str, stage_index: int, status: str
    ) -> StageLifecycleResponse:
        return self._request(
            method="POST",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/finish",
            response_type=StageLifecycleResponse,
            json_payload={"status": status},
        )

    def cloud_submit_job(
        self, *, request: CloudRunCreateRequest
    ) -> CloudRunCreateResponse:
        return self._request(
            method="POST",
            path="/api/cloud/runs",
            response_type=CloudRunCreateResponse,
            json_payload=request.to_dict(),
        )

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
        )
        if not isinstance(response_data, dict):
            raise ValueError("runtime services response must be a JSON object")
        return response_data

    def get_worker_service(
        self,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        service_id: str,
    ) -> dict[str, Any]:
        response_data = request_json(
            method="GET",
            path=f"/api/jobs/{job_id}/stages/{stage_index}/workers/{worker_id}/services/{service_id}",
            api_key=self.api_key,
            base_url=self.base_url,
        )
        if not isinstance(response_data, dict):
            raise ValueError("runtime service response must be a JSON object")
        return response_data


__all__ = [
    "MacrodataApiError",
    "MacrodataClient",
    "request_json",
    "resolve_platform_base_url",
    "sanitize_terminal_text",
    "verify_api_key",
]
