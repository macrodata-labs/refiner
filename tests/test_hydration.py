from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pytest
import refiner as mdr

from refiner.io import DataFile
from refiner.media import MediaFile, Video, hydrate_media
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.utils.cache.file_cache import get_media_cache


def _write_video(path: Path, *, fps: int = 10, frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"

        for idx in range(frames):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 10
            image[..., 1] = 255 - idx * 10
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


def test_hydrate_media_requires_video(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload_path.write_bytes(b"abc123")

    row = DictRow({"blob_uri": str(payload_path)})
    with pytest.raises(ValueError, match="hydrate_media expects a Video"):
        asyncio.run(hydrate_media("blob_uri")(row))


def test_hydrate_media_requires_decode_true(tmp_path: Path) -> None:
    video_path = tmp_path / "episode.mp4"
    _write_video(video_path)
    row = DictRow(
        {
            "video": Video(
                media=MediaFile(str(video_path)),
                video_key="cam",
                from_timestamp_s=0.0,
                to_timestamp_s=0.3,
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="hydrate_media only supports decoded Video hydration",
    ):
        asyncio.run(hydrate_media("video")(row))


def test_hydrate_media_decodes_video(tmp_path: Path) -> None:
    video_path = tmp_path / "episode.mp4"
    _write_video(video_path)
    row = DictRow(
        {
            "video": Video(
                media=MediaFile(str(video_path)),
                video_key="cam",
                from_timestamp_s=0.0,
                to_timestamp_s=0.3,
                fps=10,
            )
        }
    )

    hydrated = asyncio.run(hydrate_media("video", decode=True)(row))
    video = hydrated["video"]
    assert isinstance(video, Video)
    assert video.media.frame_count == 4
    assert video.media.width == 16
    assert video.media.height == 16


def test_hydrate_media_on_error_null(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload_path.write_bytes(b"abc123")
    row = DictRow({"blob_uri": str(payload_path)})

    hydrated = asyncio.run(hydrate_media("blob_uri", on_error="null")(row))
    assert hydrated["blob_uri"] is None


def test_map_async_hydration_preserves_row_order(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for i in range(5):
        video_path = tmp_path / f"episode-{i}.mp4"
        _write_video(video_path)
        rows.append(
            {
                "id": i,
                "video": Video(
                    media=MediaFile(str(video_path)),
                    video_key="cam",
                    from_timestamp_s=0.0,
                    to_timestamp_s=0.3,
                    fps=10,
                ),
            }
        )

    out = (
        mdr.from_items(rows)
        .map_async(hydrate_media("video", decode=True), max_in_flight=2)
        .materialize()
    )

    assert [int(r["id"]) for r in out] == [0, 1, 2, 3, 4]
    for row in out:
        video = row["video"]
        assert isinstance(video, Video)
        assert video.media.frame_count == 4


def _cached_path_once(handle: MediaFile, *, cache_name: str) -> str:
    return asyncio.run(handle.cache_file(cache_name=cache_name))


def test_media_cache_dedupes_parallel_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uri = "memory://shared-video.bin"
    seed = MediaFile(uri)
    with seed.open("wb") as f:
        f.write(b"shared")

    calls = {"count": 0}
    import refiner.pipeline.utils.cache.file_cache as media_cache

    original = media_cache._download_data_file_to_temp

    def wrapped(file: DataFile, *, cache_name: str):
        calls["count"] += 1
        time.sleep(0.05)
        return original(file=file, cache_name=cache_name)

    monkeypatch.setattr(media_cache, "_download_data_file_to_temp", wrapped)

    a = MediaFile(uri)
    b = MediaFile(uri)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(lambda: _cached_path_once(a, cache_name="parallel"))
        fut_b = pool.submit(lambda: _cached_path_once(b, cache_name="parallel"))
        path_a = fut_a.result()
        path_b = fut_b.result()

    assert calls["count"] == 1
    assert path_a == path_b
    assert os.path.exists(path_a)


def test_media_cache_with_file_cache_context_is_download_once_and_valid() -> None:
    uri = "memory://context-manager.bin"
    seed = MediaFile(uri)
    with seed.open("wb") as f:
        f.write(b"context-manager")

    calls = {"count": 0}
    cache = get_media_cache("context-manager")

    import refiner.pipeline.utils.cache.file_cache as media_cache

    original = media_cache._download_data_file_to_temp

    def wrapped(file: DataFile, *, cache_name: str):
        calls["count"] += 1
        return original(file=file, cache_name=cache_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(media_cache, "_download_data_file_to_temp", wrapped)
        data_file = DataFile.resolve(uri)
        with cache.cached(file=data_file) as path:
            assert os.path.exists(path)
            assert Path(path).read_bytes() == b"context-manager"

        with cache.cached(file=data_file) as second_path:
            assert os.path.exists(second_path)
            assert second_path == path

    assert calls["count"] == 1


def test_media_cache_file_lease_can_be_acquired_and_released_twice() -> None:
    uri = "memory://lease-same-file.bin"
    with MediaFile(uri).open("wb") as f:
        f.write(b"lease-reuse")

    cache = get_media_cache("lease-same-file")

    async def exercise() -> None:
        file = DataFile.resolve(uri)
        lease1 = await cache.acquire_file_lease(file)
        lease2 = await cache.acquire_file_lease(file)
        assert lease1.path == lease2.path
        lease1.release()
        lease2.release()

    asyncio.run(exercise())
