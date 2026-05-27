from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from refiner.inference._capabilities import ModelCapabilities
from refiner.inference._media import (
    base64_data,
    is_url,
    resolve_media_type,
    top_level_media_type,
)
from refiner.inference._message_conversion import (
    _custom_provider_data,
    _provider_options,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference._response import (
    InferenceResponse,
    _provider_metadata,
    _text_from_content,
)
from refiner.inference._transport import post_json_to_api
from refiner.inference.types import (
    InferenceWarning,
    Message,
    ProviderOptions,
    ResponseContentPart,
)

PROVIDER_OPTIONS = {
    "sendReasoning",
    "structuredOutputMode",
    "thinking",
    "disableParallelToolUse",
    "cacheControl",
    "metadata",
    "mcpServers",
    "container",
    "serviceTier",
    "toolStreaming",
    "effort",
    "taskBudget",
    "speed",
    "inferenceGeo",
    "anthropicBeta",
    "contextManagement",
}

_ENDPOINT_TIMEOUT_SECONDS = 600.0


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
                timeout=_ENDPOINT_TIMEOUT_SECONDS,
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
        return parse_response(
            response_json,
            response_headers=api_response.response_headers,
        )


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


def model_capabilities(model: str) -> ModelCapabilities:
    is_claude = "claude" in model
    return ModelCapabilities(
        images=is_claude,
        audio=False,
        video=False,
        files=is_claude,
        tools=True,
        structured_output=False,
        reasoning="3-7" in model or "4" in model or "sonnet-4" in model,
        generated_media=False,
        citations=is_claude,
    )


def build_payload(
    *,
    messages: list[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = convert_to_anthropic_payload(
        messages,
        params=_params(params),
        provider_options=provider_options,
    )
    if schema is not None:
        _apply_schema_instruction(payload, schema)
    return payload


def schema_warnings(schema: StructuredOutputSchema | None) -> list[InferenceWarning]:
    if schema is None:
        return []
    return [
        {
            "type": "unsupported-setting",
            "setting": "schema",
            "message": (
                "AnthropicEndpointProvider does not enforce schema natively; "
                "Refiner adds a JSON instruction and validates the response locally."
            ),
        }
    ]


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    if "maxOutputTokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("maxOutputTokens")
    return payload


def _apply_schema_instruction(
    payload: dict[str, Any],
    schema: StructuredOutputSchema,
) -> None:
    instruction = (
        "Return only valid JSON that matches this JSON Schema. "
        "Do not include markdown fences or extra prose.\n"
        f"{json.dumps(schema.json_schema, separators=(',', ':'))}"
    )
    existing = payload.get("system")
    if isinstance(existing, str) and existing:
        payload["system"] = f"{existing}\n\n{instruction}"
    else:
        payload["system"] = instruction


def convert_to_anthropic_payload(
    messages: Sequence[Message],
    *,
    params: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    system_parts: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            if not isinstance(content, str):
                raise ValueError("system message content must be a string")
            system_part: dict[str, Any] = {"type": "text", "text": content}
            cache_control = _anthropic_cache_control(message)
            if cache_control is not None:
                system_part["cache_control"] = cache_control
            system_parts.append(system_part)
            continue
        if role == "assistant":
            anthropic_messages.append(
                {
                    "role": "assistant",
                    "content": _convert_anthropic_assistant_content(content),
                }
            )
            continue
        if isinstance(content, str):
            anthropic_content: list[dict[str, Any]] = [
                {"type": "text", "text": content}
            ]
        else:
            anthropic_content = [
                _convert_anthropic_user_part(part, index)
                for index, part in enumerate(content)
            ]
        anthropic_messages.append({"role": "user", "content": anthropic_content})

    payload = dict(params)
    payload["messages"] = anthropic_messages
    if system_parts:
        payload["system"] = system_parts
    anthropic_options = _provider_options(provider_options, "anthropic")
    if anthropic_options:
        _apply_anthropic_options(payload, anthropic_options)
    return payload


def _convert_anthropic_user_part(part: Mapping[str, Any], index: int) -> dict[str, Any]:
    part_type = part.get("type")
    cache_control = _anthropic_cache_control(part)
    if part_type == "custom":
        payload = _custom_provider_data(part, {"anthropic"})
        if cache_control is not None:
            payload = {**payload, "cache_control": cache_control}
        return payload
    if part_type == "text":
        payload: dict[str, Any] = {"type": "text", "text": part["text"]}
        if cache_control is not None:
            payload["cache_control"] = cache_control
        return payload
    if part_type == "image":
        image_data = part.get("image")
        payload: dict[str, Any] = {
            "type": "image",
            "source": _anthropic_image_source(
                image_data,
                resolve_media_type(
                    image_data,
                    declared_media_type=part.get("mediaType"),
                    default_top_level="image",
                ),
            ),
        }
        if cache_control is not None:
            payload["cache_control"] = cache_control
        return payload
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = resolve_media_type(data, declared_media_type=part.get("mediaType"))
    top_level = top_level_media_type(media_type)
    payload: dict[str, Any]
    if top_level == "image":
        payload = {
            "type": "image",
            "source": _anthropic_image_source(data, media_type),
        }
    elif media_type == "application/pdf":
        payload = {
            "type": "document",
            "source": _anthropic_document_source(data, "application/pdf"),
            "title": _anthropic_file_title(part, index),
        }
    elif media_type == "text/plain":
        payload = {
            "type": "document",
            "source": _anthropic_text_source(data),
            "title": _anthropic_file_title(part, index),
        }
    else:
        raise ValueError(f"anthropic file part media type {media_type} is unsupported")
    anthropic_options = _provider_options(part.get("providerOptions"), "anthropic")
    if isinstance(anthropic_options.get("context"), str):
        payload["context"] = anthropic_options["context"]
    citations = anthropic_options.get("citations")
    if isinstance(citations, Mapping):
        payload["citations"] = dict(citations)
    if cache_control is not None:
        payload["cache_control"] = cache_control
    return payload


def _convert_anthropic_assistant_content(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content.strip()}]
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            parts.append({"type": "text", "text": part_text.strip()})
        elif part_type == "reasoning" and isinstance(part_text, str):
            parts.append({"type": "thinking", "thinking": part_text})
        elif part_type == "file":
            raise ValueError("anthropic assistant file parts are not supported")
        elif part_type == "custom":
            parts.append(_custom_provider_data(part, {"anthropic"}))
    return parts


def _anthropic_image_source(data: object, media_type: str) -> dict[str, Any]:
    if is_url(data):
        return {"type": "url", "url": str(data)}
    return {"type": "base64", "media_type": media_type, "data": base64_data(data)}


def _anthropic_document_source(data: object, media_type: str) -> dict[str, Any]:
    if is_url(data):
        return {"type": "url", "url": str(data)}
    return {"type": "base64", "media_type": media_type, "data": base64_data(data)}


def _anthropic_text_source(data: object) -> dict[str, Any]:
    if is_url(data):
        return {"type": "url", "url": str(data)}
    if isinstance(data, str):
        text = data
    elif isinstance(data, bytes | bytearray | memoryview):
        text = bytes(data).decode("utf-8")
    else:
        raise TypeError(
            f"text file data must be str or bytes-like, got {type(data).__name__}"
        )
    return {"type": "text", "media_type": "text/plain", "data": text}


def _anthropic_file_title(part: Mapping[str, Any], index: int) -> str | None:
    anthropic_options = _provider_options(part.get("providerOptions"), "anthropic")
    title = anthropic_options.get("title")
    if isinstance(title, str):
        return title
    filename = part.get("filename")
    if isinstance(filename, str):
        return filename
    return f"part-{index}"


def _anthropic_cache_control(part: Mapping[str, Any]) -> object:
    anthropic_options = _provider_options(part.get("providerOptions"), "anthropic")
    return anthropic_options.get("cacheControl")


def _apply_anthropic_options(
    payload: dict[str, Any], options: Mapping[str, Any]
) -> None:
    for key in (
        "container",
        "contextManagement",
        "metadata",
        "mcpServers",
        "serviceTier",
        "taskBudget",
        "thinking",
    ):
        if key in options:
            payload[key] = options[key]
    if "effort" in options:
        payload["effort"] = options["effort"]
    if "speed" in options:
        payload["speed"] = options["speed"]


def parse_response(
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
            content_parts.extend(_anthropic_sources(part.get("citations")))
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
        provider_metadata=_provider_metadata("anthropic", response_json),
    )


def _anthropic_sources(citations: object) -> list[ResponseContentPart]:
    if not isinstance(citations, Sequence) or isinstance(citations, str):
        return []
    sources: list[ResponseContentPart] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        citation = cast(Mapping[str, Any], citation)
        url = citation.get("url")
        title = citation.get("title") or citation.get("document_title")
        source: dict[str, Any] = {
            "type": "source",
            "sourceType": "url" if isinstance(url, str) else "document",
            "providerMetadata": {"anthropic": dict(citation)},
        }
        if isinstance(url, str):
            source["url"] = url
        if isinstance(title, str):
            source["title"] = title
        sources.append(cast(ResponseContentPart, source))
    return sources


__all__ = [
    "PROVIDER_OPTIONS",
    "_AnthropicEndpointClient",
    "build_payload",
    "model_capabilities",
    "parse_response",
    "schema_warnings",
]
