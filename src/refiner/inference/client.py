from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from refiner.inference._transport import post_json_to_api
from refiner.inference.types import InferenceWarning, ResponseContentPart

_OPENAI_ENDPOINT_TIMEOUT_SECONDS = 600.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    text: str
    finish_reason: str | None
    usage: Mapping[str, Any]
    response: Mapping[str, Any]
    content: Sequence[ResponseContentPart] = ()
    headers: Mapping[str, str] = field(default_factory=dict)
    warnings: Sequence[InferenceWarning] = ()

    @property
    def raw(self) -> Mapping[str, Any]:
        return self.response


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

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_normalize_openai_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        use_chat = "messages" in payload
        endpoint_path = "v1/chat/completions" if use_chat else "v1/completions"
        api_response = await post_json_to_api(
            self._ensure_client(),
            endpoint_path,
            _request_payload(payload),
            operation="generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("generation response must be a JSON object")
        return _parse_inference_response(
            response_json,
            use_chat=use_chat,
            response_headers=api_response.response_headers,
        )

    async def pooling(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "pooling",
            _request_payload(payload),
            operation="pooling",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("pooling response must be a JSON object")
        return response_json


@dataclass(slots=True)
class _GoogleEndpointClient:
    base_url: str
    model: str
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
            resolved_api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        if resolved_api_key is not None:
            headers["x-goog-api-key"] = resolved_api_key
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            f"{_google_model_path(self.model)}:generateContent",
            _request_payload(payload),
            operation="google generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("google generation response must be a JSON object")
        return _parse_google_inference_response(
            response_json,
            response_headers=api_response.response_headers,
        )


@dataclass(slots=True)
class _OpenAIResponsesClient:
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

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_normalize_openai_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "v1/responses",
            _request_payload(payload),
            operation="openai responses generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("openai responses response must be a JSON object")
        return _parse_openai_responses_response(
            response_json,
            response_headers=api_response.response_headers,
        )


@dataclass(slots=True)
class _AnthropicEndpointClient:
    base_url: str
    api_key: str | None = None
    anthropic_version: str = "2023-06-01"
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if resolved_api_key is not None:
            headers["x-api-key"] = resolved_api_key
        headers["anthropic-version"] = self.anthropic_version
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "v1/messages",
            _request_payload(payload),
            operation="anthropic generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("anthropic generation response must be a JSON object")
        return _parse_anthropic_inference_response(
            response_json,
            response_headers=api_response.response_headers,
        )


def _parse_inference_response(
    response_json: Mapping[str, Any],
    *,
    use_chat: bool,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    choices = response_json.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("generation response is missing choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("generation response choices[0] must be an object")
    content_parts: list[ResponseContentPart] = []
    if use_chat:
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("chat completion response is missing message")
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            content_parts.append({"type": "reasoning", "text": reasoning})
        content = message.get("content")
        if isinstance(content, str):
            text = content
            if text:
                content_parts.append({"type": "text", "text": text})
        elif content is None:
            logger.warning(
                "chat completion response had null message.content; returning empty text",
                extra={
                    "finish_reason": choice.get("finish_reason"),
                    "has_reasoning": isinstance(message.get("reasoning"), str),
                },
            )
            text = ""
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
            for part in parts:
                content_parts.append({"type": "text", "text": part})
        else:
            raise RuntimeError("chat completion response is missing textual content")
    else:
        text = choice.get("text")
        if not isinstance(text, str):
            raise RuntimeError("completion response is missing text")
        if text:
            content_parts.append({"type": "text", "text": text})
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
        content=content_parts,
        headers=dict(response_headers or {}),
    )


def _parse_openai_responses_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    content_parts: list[ResponseContentPart] = []
    output = response_json.get("output")
    if isinstance(output, Sequence):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, Sequence):
                    for summary_part in summary:
                        if isinstance(summary_part, Mapping) and isinstance(
                            summary_part.get("text"), str
                        ):
                            content_parts.append(
                                {"type": "reasoning", "text": summary_part["text"]}
                            )
                continue
            content = item.get("content")
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    content_parts.append({"type": "text", "text": part["text"]})
    if not _text_from_content(content_parts) and isinstance(
        response_json.get("output_text"), str
    ):
        content_parts.append({"type": "text", "text": response_json["output_text"]})
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("openai responses response is missing textual content")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    mapped_usage = {
        "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "completion_tokens": usage.get(
            "output_tokens", usage.get("completion_tokens", 0)
        ),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return InferenceResponse(
        text=text,
        finish_reason=None,
        usage=mapped_usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
    )


def _parse_anthropic_inference_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    content = response_json.get("content")
    if not isinstance(content, Sequence):
        raise RuntimeError("anthropic response is missing content")
    content_parts: list[ResponseContentPart] = []
    for part in content:
        if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
            continue
        part_type = part.get("type")
        if part_type == "text":
            content_parts.append({"type": "text", "text": part["text"]})
        elif part_type in {"thinking", "reasoning"}:
            content_parts.append({"type": "reasoning", "text": part["text"]})
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("anthropic response is missing textual content")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    mapped_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
    finish_reason = response_json.get("stop_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=mapped_usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
    )


def _parse_google_inference_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise RuntimeError("google generation response is missing candidates[0]")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise RuntimeError("google generation response candidates[0] must be an object")
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        raise RuntimeError("google generation response is missing content")
    parts = content.get("parts")
    if not isinstance(parts, Sequence):
        raise RuntimeError("google generation response is missing content.parts")
    content_parts: list[ResponseContentPart] = []
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
            continue
        if part.get("thought") is True:
            content_parts.append({"type": "reasoning", "text": part["text"]})
        else:
            content_parts.append({"type": "text", "text": part["text"]})
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("google generation response is missing textual content")
    usage_metadata = response_json.get("usageMetadata")
    usage = _google_usage(usage_metadata if isinstance(usage_metadata, Mapping) else {})
    finish_reason = candidate.get("finishReason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
    )


def _text_from_content(content: Sequence[ResponseContentPart]) -> str:
    return "".join(part["text"] for part in content if part["type"] == "text")


def _google_usage(usage_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    usage: dict[str, Any] = {}
    if "promptTokenCount" in usage_metadata:
        usage["prompt_tokens"] = usage_metadata["promptTokenCount"]
    if "candidatesTokenCount" in usage_metadata:
        usage["completion_tokens"] = usage_metadata["candidatesTokenCount"]
    if "totalTokenCount" in usage_metadata:
        usage["total_tokens"] = usage_metadata["totalTokenCount"]
    return usage


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _google_model_path(model: str) -> str:
    return model if "/" in model else f"models/{model}"


def _max_retries(payload: Mapping[str, Any]) -> int | None:
    raw = payload.get("__refiner_max_retries")
    if raw is None:
        return None
    if not isinstance(raw, int):
        raise ValueError("maxRetries must be an integer")
    return raw


def _request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    request.pop("__refiner_max_retries", None)
    return request


__all__ = ["InferenceResponse"]
