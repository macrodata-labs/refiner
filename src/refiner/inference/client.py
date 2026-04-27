from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

_OPENAI_ENDPOINT_TIMEOUT_SECONDS = 600.0
_OPENAI_ENDPOINT_MAX_RETRIES = 6
_OPENAI_ENDPOINT_RETRY_BASE_DELAY_SECONDS = 5.0
_OPENAI_ENDPOINT_RETRY_JITTER_FRACTION = 0.1


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
        for attempt in range(_OPENAI_ENDPOINT_MAX_RETRIES):
            try:
                response = await client.post(endpoint_path, json=dict(payload))
                break
            except (
                ConnectionError,
                OSError,
                asyncio.TimeoutError,
                httpx.NetworkError,
                httpx.TimeoutException,
            ) as err:
                if attempt + 1 >= _OPENAI_ENDPOINT_MAX_RETRIES:
                    message = (
                        "generation request failed after "
                        f"{_OPENAI_ENDPOINT_MAX_RETRIES} attempts: "
                        f"{type(err).__name__}: {err}"
                    )
                    raise RuntimeError(message) from err
                await asyncio.sleep(_retry_delay_seconds(attempt))
        else:
            raise RuntimeError("generation request failed without a response")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = ""
            try:
                detail = str(err.response.json())
            except ValueError:
                detail = err.response.text.strip()
            message = f"generation request failed with HTTP {err.response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from err
        response_json = response.json()
        if not isinstance(response_json, Mapping):
            raise RuntimeError("generation response must be a JSON object")
        return _parse_inference_response(
            response_json,
            use_chat=use_chat,
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
        text = _extract_chat_text(message)
        if text is None:
            raise RuntimeError(
                "chat completion response is missing textual content: "
                f"{_truncate_json(choice)}"
            )
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


_CHAT_METADATA_KEYS = frozenset(
    {
        "role",
        "type",
        "id",
        "index",
        "name",
        "url",
        "image_url",
        "audio_url",
        "tool_calls",
        "function_call",
        "logprobs",
        "annotations",
    }
)


def _extract_chat_text(content: Any) -> str | None:
    parts = _collect_chat_text_parts(content)
    if not parts:
        return None
    return "".join(parts)


def _collect_chat_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, Mapping):
        parts: list[str] = []
        preferred_keys = (
            "content",
            "text",
            "value",
            "output_text",
            "reasoning_content",
            "refusal",
        )
        for key in preferred_keys:
            if key in content:
                parts.extend(_collect_chat_text_parts(content[key]))
        if parts:
            return parts
        for key, value in content.items():
            if key in _CHAT_METADATA_KEYS:
                continue
            parts.extend(_collect_chat_text_parts(value))
        return parts
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return []
    parts: list[str] = []
    for item in content:
        parts.extend(_collect_chat_text_parts(item))
    return parts


def _truncate_json(value: Any, *, limit: int = 800) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        serialized = repr(value)
    if len(serialized) <= limit:
        return serialized
    return f"{serialized[: limit - 3]}..."


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _retry_delay_seconds(attempt: int) -> float:
    base_delay = _OPENAI_ENDPOINT_RETRY_BASE_DELAY_SECONDS * (2**attempt)
    jitter = (random.random() * 2.0 - 1.0) * _OPENAI_ENDPOINT_RETRY_JITTER_FRACTION
    return base_delay * (1.0 + jitter)


__all__ = ["InferenceResponse"]
