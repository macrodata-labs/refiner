from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

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
from refiner.inference.types import InferenceWarning, Message, ProviderOptions

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


__all__ = ["PROVIDER_OPTIONS", "build_payload", "schema_warnings"]
