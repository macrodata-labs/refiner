from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import msgspec

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
    # Drop ASCII control chars (including ESC) to avoid terminal escape injection.
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
        raise MacrodataApiError(
            status=resp.status_code, message=_http_error_message(resp)
        )

    return _decode_json_object(resp, context=path)


def parse_json_response(response_data: dict[str, Any], response_type: type[T]) -> T:
    try:
        return msgspec.convert(response_data, type=response_type, strict=True)
    except (TypeError, msgspec.ValidationError) as err:
        raise MacrodataApiError(status=200, message=str(err)) from err
