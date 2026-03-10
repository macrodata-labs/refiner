from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest

from refiner.hydration import hydrate_file
from refiner.sources.row import DictRow
from refiner.video import Video, VideoFile


def test_video_decode_true_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        Video(uri="memory://video.mp4", video_key="cam", decode=True)


def test_hydrate_file_updates_video_and_string_columns(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"abc123"
    payload_path.write_bytes(payload)
    uri = str(payload_path)

    row = DictRow(
        {
            "video": Video(uri=uri, video_key="observation.images.main"),
            "blob_uri": uri,
        }
    )
    hydrator = hydrate_file(columns=("video", "blob_uri"), max_in_flight=1)
    out = list(hydrator(row))
    out.extend(hydrator.flush())

    assert len(out) == 1
    hydrated_row = out[0]
    assert hydrated_row["blob_uri"] == payload
    hydrated_video = hydrated_row["video"]
    assert isinstance(hydrated_video, Video)
    assert isinstance(hydrated_video.file, VideoFile)
    assert hydrated_video.file.local_path is not None
    assert os.path.exists(hydrated_video.file.local_path)
    with hydrated_video.file.open("rb") as f:
        assert f.read() == payload
    assert hydrated_video.uri == uri


def test_hydrate_file_on_error_null(tmp_path: Path) -> None:
    missing_uri = str(tmp_path / "missing.bin")
    hydrator = hydrate_file(
        columns=("video", "blob_uri"),
        on_error="null",
        max_in_flight=1,
    )
    out = list(
        hydrator(
            DictRow(
                {
                    "video": Video(uri=missing_uri, video_key="cam"),
                    "blob_uri": missing_uri,
                }
            )
        )
    )
    out.extend(hydrator.flush())

    assert len(out) == 1
    hydrated = out[0]
    assert hydrated["blob_uri"] is None
    video = hydrated["video"]
    assert isinstance(video, Video)
    assert video.file is None
    assert video.bytes is None


def test_hydrate_file_skips_already_hydrated_video_file() -> None:
    row = DictRow(
        {
            "video": Video(
                uri="memory://video.mp4",
                video_key="cam",
                file=VideoFile("memory://video.mp4"),
            ),
        }
    )
    hydrator = hydrate_file(columns="video", max_in_flight=1)
    out = list(hydrator(row))
    out.extend(hydrator.flush())
    assert len(out) == 1
    assert out[0] == row


def test_hydrate_file_video_bytes_mode(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"bytes-mode"
    payload_path.write_bytes(payload)
    row = DictRow({"video": Video(uri=str(payload_path), video_key="cam")})

    hydrator = hydrate_file(columns="video", video_hydration="bytes", max_in_flight=1)
    out = list(hydrator(row))
    out.extend(hydrator.flush())
    assert len(out) == 1
    hydrated = out[0]["video"]
    assert isinstance(hydrated, Video)
    assert hydrated.bytes == payload
    assert hydrated.file is None


def test_hydrate_file_preserves_order_with_buffering(tmp_path: Path) -> None:
    payload_a = b"aaa"
    payload_b = b"bbb"
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(payload_a)
    path_b.write_bytes(payload_b)

    rows = [
        DictRow({"id": 1}),
        DictRow(
            {
                "id": 2,
                "blob_uri": str(path_a),
                "video": Video(uri=str(path_a), video_key="cam"),
            }
        ),
        DictRow(
            {
                "id": 3,
                "blob_uri": str(path_b),
                "video": Video(uri=str(path_b), video_key="cam"),
            }
        ),
    ]

    hydrator = hydrate_file(columns=("blob_uri", "video"), max_in_flight=2)
    out = []
    for row in rows:
        out.extend(hydrator(row))
    out.extend(hydrator.flush())

    assert [int(r["id"]) for r in out] == [1, 2, 3]
    assert "blob_uri" not in out[0]
    assert out[1]["blob_uri"] == payload_a
    assert out[2]["blob_uri"] == payload_b
    assert isinstance(out[1]["video"], Video)
    assert isinstance(out[1]["video"].file, VideoFile)
    assert isinstance(out[2]["video"], Video)
    assert isinstance(out[2]["video"].file, VideoFile)


def test_hydrate_file_validates_max_in_flight() -> None:
    with pytest.raises(ValueError):
        hydrate_file(columns="blob_uri", max_in_flight=0)


def test_video_file_temp_path_cleans_on_gc(tmp_path: Path) -> None:
    src = tmp_path / "video.bin"
    src.write_bytes(b"0123456789")

    handle = VideoFile(str(src))
    local_path = handle.to_local_path(suffix=".bin")
    assert os.path.exists(local_path)
    assert Path(local_path).read_bytes() == b"0123456789"

    del handle
    gc.collect()

    assert not os.path.exists(local_path)
