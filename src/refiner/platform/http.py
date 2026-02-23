from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


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
        text = resp.text.strip()
        return text or resp.reason_phrase
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("message")
        if isinstance(message, str) and message:
            return message
    text = resp.text.strip()
    return text or resp.reason_phrase


def verify_api_key(
    base_url: str, api_key: str, timeout_s: float = 10.0
) -> dict[str, Any]:
    """Validate an API key and return the platform `/api/me` JSON payload."""
    endpoint = f"{base_url}/api/me"
    try:
        resp = httpx.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )
    except httpx.RequestError as err:
        raise MacrodataApiError(status=0, message=str(err)) from err

    if resp.is_error:
        raise MacrodataApiError(
            status=resp.status_code, message=_http_error_message(resp)
        )

    return _decode_json_object(resp, context="/api/me")
