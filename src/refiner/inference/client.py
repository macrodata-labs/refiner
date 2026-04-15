from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

_OPENAI_ENDPOINT_TIMEOUT_SECONDS = 600.0
_OPENAI_ENDPOINT_MAX_RETRIES = 6
_OPENAI_ENDPOINT_RETRY_BASE_DELAY_SECONDS = 1.0
_OPENAI_ENDPOINT_RETRYABLE_STATUS_CODES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    text: str
    finish_reason: str | None
    usage: Mapping[str, Any]
    response: Mapping[str, Any]


@dataclass(slots=True)
class _OpenAIEndpointClient:
    base_url: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("OPENAI_API_KEY")
        if resolved_api_key is not None:
            headers["Authorization"] = f"Bearer {resolved_api_key}"
        self._resolved_headers = headers

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        use_chat = "messages" in payload
        endpoint_path = "v1/chat/completions" if use_chat else "v1/completions"
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_normalize_openai_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        last_error: Exception | None = None
        for attempt in range(_OPENAI_ENDPOINT_MAX_RETRIES):
            try:
                response = await client.post(endpoint_path, json=dict(payload))
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                if (
                    attempt + 1 < _OPENAI_ENDPOINT_MAX_RETRIES
                    and _is_retryable_http_status(err.response.status_code)
                ):
                    await _sleep_before_retry(attempt)
                    last_error = err
                    continue
                detail = ""
                try:
                    detail = str(err.response.json())
                except ValueError:
                    detail = err.response.text.strip()
                message = (
                    f"generation request failed with HTTP {err.response.status_code}"
                )
                if detail:
                    message = f"{message}: {detail}"
                raise RuntimeError(message) from err
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.TimeoutException,
                OSError,
            ) as err:
                if attempt + 1 < _OPENAI_ENDPOINT_MAX_RETRIES:
                    await _sleep_before_retry(attempt)
                    last_error = err
                    continue
                raise RuntimeError(
                    f"generation request failed after retries: {type(err).__name__}: {err}"
                ) from err
            else:
                response_json = response.json()
                if not isinstance(response_json, Mapping):
                    raise RuntimeError("generation response must be a JSON object")
                return _parse_inference_response(
                    response_json,
                    use_chat=use_chat,
                )
        raise RuntimeError(
            f"generation request failed after retries: {type(last_error).__name__ if last_error is not None else 'unknown'}"
        )


def _parse_inference_response(
    response_json: Mapping[str, Any], *, use_chat: bool
) -> InferenceResponse:
    choices = response_json.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("generation response is missing choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("generation response choices[0] must be an object")
    if use_chat:
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("chat completion response is missing message")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, Sequence):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if not parts:
                raise RuntimeError(
                    "chat completion response is missing textual content"
                )
            text = "".join(parts)
        else:
            raise RuntimeError("chat completion response is missing textual content")
    else:
        text = choice.get("text")
        if not isinstance(text, str):
            raise RuntimeError("completion response is missing text")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        response=response_json,
    )


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _is_retryable_http_status(status_code: int) -> bool:
    return int(status_code) in _OPENAI_ENDPOINT_RETRYABLE_STATUS_CODES


async def _sleep_before_retry(attempt: int) -> None:
    delay_s = _OPENAI_ENDPOINT_RETRY_BASE_DELAY_SECONDS * (2**attempt)
    await asyncio.sleep(delay_s)


__all__ = ["InferenceResponse"]
