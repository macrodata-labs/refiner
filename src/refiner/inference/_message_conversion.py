from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from refiner.inference.types import Message

_DATA_URL_PATTERN = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_WILDCARD_MEDIA_SUFFIX = "/*"


def convert_to_openai_chat_messages(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role in {"system", "assistant"}:
            converted.append({"role": role, "content": content})
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
                    "parts": [{"text": content}] if content else [],
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
        if role in {"system", "assistant"}:
            input_items.append({"role": role, "content": content})
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
            if not isinstance(content, str):
                raise ValueError("assistant message content must be a string")
            anthropic_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": content.strip()}],
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
    if part_type == "text":
        return {"type": "text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _openai_image_url_part(
            image_data,
            media_type=_resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = _resolve_media_type(
        data,
        declared_media_type=part.get("mediaType"),
    )
    top_level = _top_level_media_type(media_type)
    if top_level == "image":
        return _openai_image_url_part(
            data,
            media_type=media_type,
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if top_level == "audio":
        if _is_url(data):
            raise ValueError("openai audio file parts with URLs are not supported")
        full_media_type = _resolve_media_type(
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
                "data": _base64_data(data),
                "format": audio_format,
            },
        }
    full_media_type = _resolve_media_type(data, declared_media_type=media_type)
    if full_media_type != "application/pdf":
        raise ValueError(
            f"openai file part media type {full_media_type} is unsupported"
        )
    if _is_url(data):
        raise ValueError("openai PDF file parts with URLs are not supported")
    return {
        "type": "file",
        "file": {
            "filename": part.get("filename", f"part-{index}.pdf"),
            "file_data": f"data:application/pdf;base64,{_base64_data(data)}",
        },
    }


def _convert_openai_responses_part(
    part: Mapping[str, Any], index: int
) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "input_text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": _data_or_url(
                image_data,
                _resolve_media_type(
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
    media_type = _resolve_media_type(data, declared_media_type=part.get("mediaType"))
    if _top_level_media_type(media_type) == "image":
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": _data_or_url(data, media_type),
        }
        detail = _provider_option(part, "openai", "imageDetail")
        if detail is not None:
            image_part["detail"] = detail
        return image_part
    if _is_url(data):
        return {"type": "input_file", "file_url": str(data)}
    full_media_type = _resolve_media_type(data, declared_media_type=media_type)
    return {
        "type": "input_file",
        "filename": part.get(
            "filename",
            f"part-{index}.pdf"
            if full_media_type == "application/pdf"
            else f"part-{index}",
        ),
        "file_data": f"data:{full_media_type};base64,{_base64_data(data)}",
    }


def _openai_image_url_part(
    data: object,
    *,
    media_type: object,
    detail: object,
) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": _data_or_url(data, str(media_type))}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _convert_google_user_part(part: Mapping[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        return {"text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _google_file_part(
            image_data,
            _resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
        )
    if part_type == "file":
        file_data = part.get("data")
        return _google_file_part(
            file_data,
            _resolve_media_type(file_data, declared_media_type=part.get("mediaType")),
        )
    raise ValueError(f"unsupported content part type {part_type!r}")


def _convert_anthropic_user_part(part: Mapping[str, Any], index: int) -> dict[str, Any]:
    part_type = part.get("type")
    cache_control = _anthropic_cache_control(part)
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
                _resolve_media_type(
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
    media_type = _resolve_media_type(data, declared_media_type=part.get("mediaType"))
    top_level = _top_level_media_type(media_type)
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


def _google_file_part(data: object, media_type: str) -> dict[str, Any]:
    parsed = _parse_data_url(data)
    if parsed is not None:
        parsed_media_type, parsed_data = parsed
        return {"inlineData": {"mimeType": parsed_media_type, "data": parsed_data}}
    if _is_url(data):
        return {"fileData": {"mimeType": media_type, "fileUri": str(data)}}
    return {"inlineData": {"mimeType": media_type, "data": _base64_data(data)}}


def _anthropic_image_source(data: object, media_type: str) -> dict[str, Any]:
    if _is_url(data):
        return {"type": "url", "url": str(data)}
    return {"type": "base64", "media_type": media_type, "data": _base64_data(data)}


def _anthropic_document_source(data: object, media_type: str) -> dict[str, Any]:
    if _is_url(data):
        return {"type": "url", "url": str(data)}
    return {"type": "base64", "media_type": media_type, "data": _base64_data(data)}


def _anthropic_text_source(data: object) -> dict[str, Any]:
    if _is_url(data):
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
        "responseModalities",
        "thinkingConfig",
        "audioTimestamp",
        "mediaResolution",
        "imageConfig",
    }
    generation_config = payload.setdefault("generationConfig", {})
    if isinstance(generation_config, dict):
        for key in generation_keys:
            if key in options:
                generation_config[key] = options[key]
    for key in ("safetySettings", "cachedContent", "labels", "serviceTier"):
        if key in options:
            payload[key] = options[key]


def _apply_anthropic_options(
    payload: dict[str, Any], options: Mapping[str, Any]
) -> None:
    for key in (
        "thinking",
        "metadata",
        "contextManagement",
        "container",
        "mcpServers",
        "service_tier",
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


def _top_level_media_type(media_type: str) -> str:
    return media_type.split("/", 1)[0]


def _resolve_media_type(
    data: object,
    *,
    declared_media_type: object,
    default_top_level: str | None = None,
) -> str:
    parsed = _parse_data_url(data)
    if parsed is not None:
        return parsed[0]
    media_type = _declared_media_type(declared_media_type)
    if media_type is not None and _is_full_media_type(media_type):
        return media_type
    detected = _detect_media_type(data)
    if detected is not None:
        if media_type is None:
            return detected
        if _top_level_media_type(detected) == _top_level_media_type(media_type):
            return detected
    if media_type is not None:
        if media_type.endswith(_WILDCARD_MEDIA_SUFFIX):
            return media_type.removesuffix(_WILDCARD_MEDIA_SUFFIX)
        return media_type
    if default_top_level == "image":
        return "image/png"
    return "application/octet-stream"


def _declared_media_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    media_type = value.strip()
    if not media_type:
        return None
    return media_type


def _is_full_media_type(media_type: str | None) -> bool:
    if media_type is None:
        return False
    if media_type.endswith(_WILDCARD_MEDIA_SUFFIX):
        return False
    top_level, separator, subtype = media_type.partition("/")
    return bool(top_level and separator and subtype)


def _detect_media_type(data: object) -> str | None:
    if not isinstance(data, bytes | bytearray | memoryview):
        return None
    raw = bytes(data)
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"%PDF-"):
        return "application/pdf"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        return "audio/wav"
    if raw.startswith(b"ID3") or _looks_like_mp3_frame(raw):
        return "audio/mpeg"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12]
        if brand in {b"M4A ", b"M4B ", b"mp42", b"isom", b"iso2"}:
            return "video/mp4"
    if raw.startswith(b"OggS"):
        return "audio/ogg"
    return None


def _looks_like_mp3_frame(raw: bytes) -> bool:
    return len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0


def _data_or_url(data: object, media_type: str) -> str:
    parsed = _parse_data_url(data)
    if parsed is not None:
        return str(data)
    if _is_url(data):
        return str(data)
    return f"data:{media_type};base64,{_base64_data(data)}"


def _base64_data(data: object) -> str:
    if isinstance(data, str):
        parsed = _parse_data_url(data)
        if parsed is not None:
            return parsed[1]
        return data
    if isinstance(data, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(data)).decode("ascii")
    raise TypeError(f"file data must be str or bytes-like, got {type(data).__name__}")


def _parse_data_url(data: object) -> tuple[str, str] | None:
    if not isinstance(data, str):
        return None
    match = _DATA_URL_PATTERN.match(data)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _is_url(data: object) -> bool:
    return isinstance(data, str) and (
        data.startswith("http://") or data.startswith("https://")
    )


__all__ = [
    "convert_to_anthropic_payload",
    "convert_to_google_payload",
    "convert_to_openai_chat_messages",
    "convert_to_openai_responses_input",
]
