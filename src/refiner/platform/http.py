from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


def _sanitize_terminal_text(value: str) -> str:
    # Drop ASCII control chars (including ESC) to avoid terminal escape injection.
    return "".join(ch for ch in value if ch >= " " and ch != "\x7f")


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
            return _sanitize_terminal_text(resp.reason_phrase or "HTTP error")
        if not text:
            return _sanitize_terminal_text(resp.reason_phrase or "HTTP error")
        one_line = " ".join(text.split())
        if len(one_line) > 200:
            one_line = f"{one_line[:197]}..."
        return _sanitize_terminal_text(one_line)
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("message")
        if isinstance(message, str) and message:
            return _sanitize_terminal_text(message)
    return _sanitize_terminal_text(resp.reason_phrase or "HTTP error")


def request_json(
    *,
    method: str,
    path: str,
    api_key: str,
    base_url: str,
    json_payload: dict[str, Any] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = httpx.request(
            method,
            url,
            headers={"Authorization": f"Bearer {api_key}"},
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


def verify_api_key(
    base_url: str, api_key: str, timeout_s: float = 10.0
) -> dict[str, Any]:
    """Validate an API key and return the platform `/api/me` JSON payload."""
    return request_json(
        method="GET",
        path="/api/me",
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
    )
