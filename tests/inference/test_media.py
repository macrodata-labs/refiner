from __future__ import annotations

import pytest

from refiner.inference.internal.media import (
    base64_data,
    data_or_url,
    detect_media_type,
    parse_data_url,
    resolve_media_type,
)


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (b"\x89PNG\r\n\x1a\npayload", "image/png"),
        (b"\xff\xd8\xffpayload", "image/jpeg"),
        (b"GIF89apayload", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPpayload", "image/webp"),
        (b"%PDF-1.7\npayload", "application/pdf"),
        (b"RIFF\x00\x00\x00\x00WAVEpayload", "audio/wav"),
        (b"ID3payload", "audio/mpeg"),
        (b"\x00\x00\x00\x18ftypisompayload", "video/mp4"),
        (b"OggSpayload", "audio/ogg"),
    ],
)
def test_detect_media_type_from_byte_signatures(
    data: bytes,
    media_type: str,
) -> None:
    assert detect_media_type(data) == media_type


def test_resolve_media_type_prefers_data_url_media_type() -> None:
    assert (
        resolve_media_type(
            "data:image/jpeg;base64,aGVsbG8=",
            declared_media_type="image/png",
        )
        == "image/jpeg"
    )


def test_resolve_media_type_refines_top_level_declaration_from_bytes() -> None:
    assert (
        resolve_media_type(
            b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00",
            declared_media_type="video",
        )
        == "video/mp4"
    )
    assert (
        resolve_media_type(
            b"\x89PNG\r\n\x1a\npayload",
            declared_media_type="image/*",
        )
        == "image/png"
    )


def test_resolve_media_type_falls_back_to_declared_or_default_type() -> None:
    assert resolve_media_type(b"unknown", declared_media_type="video") == "video"
    assert (
        resolve_media_type(
            b"unknown",
            declared_media_type=None,
            default_top_level="image",
        )
        == "image/png"
    )
    assert (
        resolve_media_type(b"unknown", declared_media_type=None)
        == "application/octet-stream"
    )


def test_data_url_helpers_preserve_urls_and_encode_bytes() -> None:
    assert parse_data_url("data:text/plain;base64,aGVsbG8=") == (
        "text/plain",
        "aGVsbG8=",
    )
    assert base64_data(b"hello") == "aGVsbG8="
    assert data_or_url("https://example.com/image.png", "image/png") == (
        "https://example.com/image.png"
    )
    assert data_or_url(b"hello", "text/plain") == "data:text/plain;base64,aGVsbG8="
