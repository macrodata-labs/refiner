from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from refiner.inference._media import (
    base64_data,
    data_or_url,
    is_url,
    parse_data_url,
    resolve_media_type,
    top_level_media_type,
)
from refiner.inference.types import Message


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


def _custom_provider_data(
    part: Mapping[str, Any], provider_aliases: set[str]
) -> dict[str, Any]:
    provider = part.get("provider")
    if not isinstance(provider, str) or provider not in provider_aliases:
        aliases = ", ".join(sorted(provider_aliases))
        raise ValueError(
            f"custom content part provider must be one of {aliases}; got {provider!r}"
        )
    data = part.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("custom content part data must be an object")
    return dict(data)


def _google_file_part(data: object, media_type: str) -> dict[str, Any]:
    parsed = parse_data_url(data)
    if parsed is not None:
        parsed_media_type, parsed_data = parsed
        return {"inlineData": {"mimeType": parsed_media_type, "data": parsed_data}}
    if is_url(data):
        return {"fileData": {"mimeType": media_type, "fileUri": str(data)}}
    return {"inlineData": {"mimeType": media_type, "data": base64_data(data)}}


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


def _provider_option(part: Mapping[str, Any], provider: str, key: str) -> object:
    provider_options = part.get("providerOptions")
    if not isinstance(provider_options, Mapping):
        return None
    options = cast(Mapping[str, Any], provider_options).get(provider)
    if not isinstance(options, Mapping):
        return None
    return options.get(key)


def _provider_options(provider_options: object, provider: str) -> Mapping[str, Any]:
    if not isinstance(provider_options, Mapping):
        return {}
    options = dict(provider_options).get(provider)
    if not isinstance(options, Mapping):
        return {}
    return options


__all__ = [
    "convert_to_anthropic_payload",
    "convert_to_google_payload",
    "convert_to_openai_chat_messages",
    "convert_to_openai_responses_input",
]
