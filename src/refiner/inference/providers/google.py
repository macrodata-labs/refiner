from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from refiner.inference._media import (
    base64_data,
    is_url,
    parse_data_url,
    resolve_media_type,
)
from refiner.inference._message_conversion import (
    _custom_provider_data,
    _provider_options,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference.types import Message, ProviderOptions

PROVIDER_OPTIONS = {
    "apiClient",
    "responseModalities",
    "responseMimeType",
    "responseSchema",
    "thinkingConfig",
    "cachedContent",
    "structuredOutputs",
    "safetySettings",
    "threshold",
    "audioTimestamp",
    "labels",
    "mediaResolution",
    "imageConfig",
    "retrievalConfig",
    "speechConfig",
    "systemInstruction",
    "streamFunctionCallArguments",
    "serviceTier",
    "sharedRequestType",
    "requestType",
}


def build_payload(
    *,
    messages: list[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = convert_to_google_payload(
        messages,
        generation_config=_generation_config(params),
        provider_options=provider_options,
    )
    if schema is not None:
        generation_config = dict(payload.get("generationConfig", {}))
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema.json_schema
        payload["generationConfig"] = generation_config
    return payload


def _generation_config(params: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(params)
    aliases = {
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "stop_sequences": "stopSequences",
    }
    for source, target in aliases.items():
        if source in config and target not in config:
            config[target] = config.pop(source)
    return config


def convert_to_google_payload(
    messages: Sequence[Message],
    *,
    generation_config: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    system_messages_allowed = True

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            if not isinstance(content, str):
                raise ValueError("system message content must be a string")
            if not system_messages_allowed:
                raise ValueError(
                    "google system messages are only supported before user messages"
                )
            system_parts.append({"text": content})
            continue
        system_messages_allowed = False
        if role == "assistant":
            contents.append(
                {
                    "role": "model",
                    "parts": _convert_google_assistant_content(content),
                }
            )
            continue
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = [_convert_google_user_part(part) for part in content]
        contents.append({"role": "user", "parts": parts})

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if generation_config:
        payload["generationConfig"] = dict(generation_config)
    google_options = _provider_options(provider_options, "google")
    if google_options:
        _apply_google_options(payload, google_options)
    return payload


def _convert_google_user_part(part: Mapping[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"google"})
    if part_type == "text":
        return {"text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _google_file_part(
            image_data,
            resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
        )
    if part_type == "file":
        file_data = part.get("data")
        return _google_file_part(
            file_data,
            resolve_media_type(file_data, declared_media_type=part.get("mediaType")),
        )
    raise ValueError(f"unsupported content part type {part_type!r}")


def _convert_google_assistant_content(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
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
            if part_text:
                parts.append({"text": part_text})
        elif part_type == "reasoning" and isinstance(part_text, str):
            if part_text:
                parts.append({"text": part_text, "thought": True})
        elif part_type == "file":
            data = part.get("data")
            if is_url(data):
                raise ValueError("google assistant file URLs are not supported")
            media_type = resolve_media_type(
                data,
                declared_media_type=part.get("mediaType"),
            )
            parts.append(
                {
                    "inlineData": {
                        "mimeType": media_type,
                        "data": base64_data(data),
                    }
                }
            )
        elif part_type == "custom":
            parts.append(_custom_provider_data(part, {"google"}))
    return parts


def _google_file_part(data: object, media_type: str) -> dict[str, Any]:
    parsed = parse_data_url(data)
    if parsed is not None:
        parsed_media_type, parsed_data = parsed
        return {"inlineData": {"mimeType": parsed_media_type, "data": parsed_data}}
    if is_url(data):
        return {"fileData": {"mimeType": media_type, "fileUri": str(data)}}
    return {"inlineData": {"mimeType": media_type, "data": base64_data(data)}}


def _apply_google_options(payload: dict[str, Any], options: Mapping[str, Any]) -> None:
    generation_keys = {
        "audioTimestamp",
        "imageConfig",
        "mediaResolution",
        "responseMimeType",
        "responseModalities",
        "responseSchema",
        "speechConfig",
        "thinkingConfig",
    }
    generation_config = payload.setdefault("generationConfig", {})
    if isinstance(generation_config, dict):
        for key in generation_keys:
            if key in options:
                generation_config[key] = options[key]
    for key in (
        "cachedContent",
        "labels",
        "retrievalConfig",
        "safetySettings",
        "serviceTier",
        "systemInstruction",
    ):
        if key in options:
            payload[key] = options[key]


__all__ = ["PROVIDER_OPTIONS", "build_payload"]
