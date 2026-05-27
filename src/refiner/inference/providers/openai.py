from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from refiner.inference._media import (
    base64_data,
    data_or_url,
    is_url,
    resolve_media_type,
    top_level_media_type,
)
from refiner.inference._message_conversion import (
    _custom_provider_data,
    _provider_option,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference.types import Message, ProviderOptions

CHAT_PROVIDER_OPTIONS = {
    "audio",
    "background",
    "logitBias",
    "logprobs",
    "modalities",
    "parallelToolCalls",
    "user",
    "responseFormat",
    "reasoningEffort",
    "maxCompletionTokens",
    "store",
    "metadata",
    "prediction",
    "serviceTier",
    "reasoningSummary",
    "textVerbosity",
    "promptCacheKey",
    "promptCacheRetention",
    "safetyIdentifier",
    "text",
    "topLogprobs",
    "webSearchOptions",
}

RESPONSES_PROVIDER_OPTIONS = {
    *CHAT_PROVIDER_OPTIONS,
    "conversation",
    "include",
    "instructions",
    "maxToolCalls",
    "previousResponseId",
    "truncation",
    "contextManagement",
}


def build_chat_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["messages"] = convert_to_openai_chat_messages(messages)
    elif schema is None:
        payload["prompt"] = prompt
    else:
        payload["messages"] = [{"role": "user", "content": prompt or ""}]
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": _json_schema(schema),
        }
    if provider_options is not None:
        payload["providerOptions"] = provider_options
    return payload


def build_responses_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["input"] = convert_to_openai_responses_input(messages)
    else:
        payload["input"] = prompt
    _apply_responses_options(payload, provider_options)
    _normalize_responses_params(payload)
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        text = dict(payload.get("text", {}))
        text["format"] = _response_format(schema)
        payload["text"] = text
    return payload


def _apply_responses_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    passthrough = {
        "conversation",
        "background",
        "include",
        "instructions",
        "logprobs",
        "maxToolCalls",
        "metadata",
        "parallelToolCalls",
        "previousResponseId",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "serviceTier",
        "store",
        "truncation",
        "user",
        "contextManagement",
        "text",
        "topLogprobs",
        "webSearchOptions",
    }
    for key in passthrough:
        if key in openai_options:
            payload[key] = openai_options[key]


def _normalize_responses_params(payload: dict[str, Any]) -> None:
    aliases = {
        "max_tokens": "max_output_tokens",
        "maxCompletionTokens": "max_output_tokens",
    }
    for source, target in aliases.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)


def _normalize_reasoning_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    for key in (
        "logitBias",
        "logprobs",
        "topLogprobs",
        "parallelToolCalls",
        "user",
        "maxCompletionTokens",
        "modalities",
        "audio",
        "store",
        "metadata",
        "prediction",
        "serviceTier",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "webSearchOptions",
        "responseFormat",
    ):
        if key in openai_options:
            payload[key] = openai_options[key]
    reasoning: dict[str, Any] = {}
    if "reasoningEffort" in openai_options:
        reasoning["effort"] = openai_options["reasoningEffort"]
    if "reasoningSummary" in openai_options:
        reasoning["summary"] = openai_options["reasoningSummary"]
    if reasoning:
        payload["reasoning"] = {**dict(payload.get("reasoning", {})), **reasoning}


def _normalize_text_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    if "textVerbosity" in openai_options:
        text = dict(payload.get("text", {}))
        text["verbosity"] = openai_options["textVerbosity"]
        payload["text"] = text


def _response_format(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


def _json_schema(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


def convert_to_openai_chat_messages(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append({"role": role, "content": content})
            continue
        if role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": _convert_openai_chat_assistant_content(content),
                }
            )
            continue
        if isinstance(content, str):
            converted.append({"role": "user", "content": content})
            continue
        if len(content) == 1 and content[0]["type"] == "text":
            converted.append({"role": "user", "content": content[0]["text"]})
            continue
        converted.append(
            {
                "role": "user",
                "content": [
                    _convert_openai_user_part(part, index)
                    for index, part in enumerate(content)
                ],
            }
        )
    return converted


def convert_to_openai_responses_input(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            input_items.append({"role": role, "content": content})
            continue
        if role == "assistant":
            input_items.extend(_convert_openai_responses_assistant_content(content))
            continue
        if isinstance(content, str):
            input_items.append({"role": "user", "content": content})
            continue
        input_items.append(
            {
                "role": "user",
                "content": [
                    _convert_openai_responses_part(part, index)
                    for index, part in enumerate(content)
                ],
            }
        )
    return input_items


def _convert_openai_user_part(part: Mapping[str, Any], index: int) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"openai", "openai-chat"})
    if part_type == "text":
        return {"type": "text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _openai_image_url_part(
            image_data,
            media_type=resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = resolve_media_type(
        data,
        declared_media_type=part.get("mediaType"),
    )
    top_level = top_level_media_type(media_type)
    if top_level == "image":
        return _openai_image_url_part(
            data,
            media_type=media_type,
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if top_level == "audio":
        if is_url(data):
            raise ValueError("openai audio file parts with URLs are not supported")
        full_media_type = resolve_media_type(
            data,
            declared_media_type=media_type,
            default_top_level="audio",
        )
        if full_media_type in {"audio/wav", "audio/x-wav"}:
            audio_format = "wav"
        elif full_media_type in {"audio/mp3", "audio/mpeg"}:
            audio_format = "mp3"
        else:
            raise ValueError(
                f"openai audio content parts with media type {full_media_type} "
                "are not supported"
            )
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64_data(data),
                "format": audio_format,
            },
        }
    full_media_type = resolve_media_type(data, declared_media_type=media_type)
    if full_media_type != "application/pdf":
        raise ValueError(
            f"openai file part media type {full_media_type} is unsupported"
        )
    if is_url(data):
        raise ValueError("openai PDF file parts with URLs are not supported")
    return {
        "type": "file",
        "file": {
            "filename": part.get("filename", f"part-{index}.pdf"),
            "file_data": f"data:application/pdf;base64,{base64_data(data)}",
        },
    }


def _convert_openai_responses_part(
    part: Mapping[str, Any], index: int
) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"openai", "openai-responses"})
    if part_type == "text":
        return {"type": "input_text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": data_or_url(
                image_data,
                resolve_media_type(
                    image_data,
                    declared_media_type=part.get("mediaType"),
                    default_top_level="image",
                ),
            ),
        }
        detail = _provider_option(part, "openai", "imageDetail")
        if detail is not None:
            image_part["detail"] = detail
        return image_part
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = resolve_media_type(data, declared_media_type=part.get("mediaType"))
    if top_level_media_type(media_type) == "image":
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": data_or_url(data, media_type),
        }
        detail = _provider_option(part, "openai", "imageDetail")
        if detail is not None:
            image_part["detail"] = detail
        return image_part
    if is_url(data):
        return {"type": "input_file", "file_url": str(data)}
    full_media_type = resolve_media_type(data, declared_media_type=media_type)
    return {
        "type": "input_file",
        "filename": part.get(
            "filename",
            f"part-{index}.pdf"
            if full_media_type == "application/pdf"
            else f"part-{index}",
        ),
        "file_data": f"data:{full_media_type};base64,{base64_data(data)}",
    }


def _openai_image_url_part(
    data: object,
    *,
    media_type: object,
    detail: object,
) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": data_or_url(data, str(media_type))}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _convert_openai_chat_assistant_content(
    content: object,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    text_parts: list[str] = []
    converted_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            text_parts.append(part_text)
            converted_parts.append({"type": "text", "text": part_text})
        elif part_type == "custom":
            converted_parts.append(
                _custom_provider_data(part, {"openai", "openai-chat"})
            )
        elif part_type in {"file", "reasoning"}:
            raise ValueError(
                f"openai chat assistant {part_type} parts are not supported"
            )
    if any(part.get("type") != "text" for part in converted_parts):
        return converted_parts
    return "".join(text_parts)


def _convert_openai_responses_assistant_content(
    content: object,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    input_items: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            text_parts.append(part_text)
        elif part_type == "reasoning" and isinstance(part_text, str):
            input_items.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": part_text}],
                }
            )
        elif part_type == "file":
            raise ValueError("openai responses assistant file parts are not supported")
        elif part_type == "custom":
            input_items.append(
                _custom_provider_data(part, {"openai", "openai-responses"})
            )
    if text_parts:
        input_items.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(text_parts)}],
            }
        )
    return input_items


__all__ = [
    "CHAT_PROVIDER_OPTIONS",
    "RESPONSES_PROVIDER_OPTIONS",
    "build_chat_payload",
    "build_responses_payload",
]
