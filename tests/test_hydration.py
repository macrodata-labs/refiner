from __future__ import annotations

import asyncio
import os
from pathlib import Path

import refiner as mdr
import pytest

from refiner.media import MediaFile, Video, hydrate_media
from refiner.sources.row import DictRow


def test_video_decode_true_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        Video(media=MediaFile("memory://video.mp4"), video_key="cam", decode=True)


def test_hydrate_media_turns_string_into_media_file(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"abc123"
    payload_path.write_bytes(payload)

    row = DictRow({"blob_uri": str(payload_path)})
    hydrated = asyncio.run(hydrate_media("blob_uri", mode="file")(row))
    value = hydrated["blob_uri"]

    assert isinstance(value, MediaFile)
    assert value.local_path is not None
    assert os.path.exists(value.local_path)
    with value.open("rb") as f:
        assert f.read() == payload


def test_hydrate_media_hydrates_video_wrapper_bytes(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"bytes-mode"
    payload_path.write_bytes(payload)
    row = DictRow({"video": Video(media=MediaFile(str(payload_path)), video_key="cam")})

    hydrated = asyncio.run(hydrate_media("video", mode="bytes")(row))
    video = hydrated["video"]
    assert isinstance(video, Video)
    assert video.media.bytes_cache == payload


def test_hydrate_media_on_error_null(tmp_path: Path) -> None:
    missing_uri = str(tmp_path / "missing.bin")
    row = DictRow({"blob_uri": missing_uri})

    hydrated = asyncio.run(hydrate_media("blob_uri", on_error="null")(row))
    assert hydrated["blob_uri"] is None


def test_map_async_hydration_preserves_row_order(tmp_path: Path) -> None:
    payloads: dict[int, bytes] = {}
    rows: list[dict[str, object]] = []
    for i in range(5):
        payload = f"row-{i}".encode()
        p = tmp_path / f"item-{i}.bin"
        p.write_bytes(payload)
        payloads[i] = payload
        rows.append({"id": i, "blob_uri": str(p)})

    out = (
        mdr.from_items(rows)
        .map_async(hydrate_media("blob_uri", mode="bytes"), max_in_flight=2)
        .materialize()
    )

    assert [int(r["id"]) for r in out] == [0, 1, 2, 3, 4]
    for row in out:
        idx = int(row["id"])
        media = row["blob_uri"]
        assert isinstance(media, MediaFile)
        assert media.bytes_cache == payloads[idx]


def test_media_file_temp_path_cleans_on_cleanup() -> None:
    handle = MediaFile("memory://video.bin")
    with handle.open("wb") as f:
        f.write(b"0123456789")

    local_path = handle.cache_locally(suffix=".bin")
    assert os.path.exists(local_path)
    assert Path(local_path).read_bytes() == b"0123456789"

    handle.cleanup()

    assert not os.path.exists(local_path)
