from __future__ import annotations

import base64
import re

_DATA_URL_PATTERN = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_WILDCARD_MEDIA_SUFFIX = "/*"


def top_level_media_type(media_type: str) -> str:
    return media_type.split("/", 1)[0]


def resolve_media_type(
    data: object,
    *,
    declared_media_type: object,
    default_top_level: str | None = None,
) -> str:
    parsed = parse_data_url(data)
    if parsed is not None:
        return parsed[0]
    media_type = _declared_media_type(declared_media_type)
    if media_type is not None and _is_full_media_type(media_type):
        return media_type
    detected = detect_media_type(data)
    if detected is not None:
        if media_type is None:
            return detected
        if top_level_media_type(detected) == top_level_media_type(media_type):
            return detected
    if media_type is not None:
        if media_type.endswith(_WILDCARD_MEDIA_SUFFIX):
            return media_type.removesuffix(_WILDCARD_MEDIA_SUFFIX)
        return media_type
    if default_top_level == "image":
        return "image/png"
    return "application/octet-stream"


def detect_media_type(data: object) -> str | None:
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


def data_or_url(data: object, media_type: str) -> str:
    parsed = parse_data_url(data)
    if parsed is not None:
        return str(data)
    if is_url(data):
        return str(data)
    return f"data:{media_type};base64,{base64_data(data)}"


def base64_data(data: object) -> str:
    if isinstance(data, str):
        parsed = parse_data_url(data)
        if parsed is not None:
            return parsed[1]
        return data
    if isinstance(data, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(data)).decode("ascii")
    raise TypeError(f"file data must be str or bytes-like, got {type(data).__name__}")


def parse_data_url(data: object) -> tuple[str, str] | None:
    if not isinstance(data, str):
        return None
    match = _DATA_URL_PATTERN.match(data)
    if match is None:
        return None
    return match.group(1), match.group(2)


def is_url(data: object) -> bool:
    return isinstance(data, str) and (
        data.startswith("http://") or data.startswith("https://")
    )


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


def _looks_like_mp3_frame(raw: bytes) -> bool:
    return len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0


__all__ = [
    "base64_data",
    "data_or_url",
    "detect_media_type",
    "is_url",
    "parse_data_url",
    "resolve_media_type",
    "top_level_media_type",
]
