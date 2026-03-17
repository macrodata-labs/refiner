from __future__ import annotations

import asyncio
import os
from pathlib import Path

import av
import numpy as np

import refiner as mdr
from refiner.io import DataFile
from refiner.media import DecodedVideo, VideoFile, hydrate_video
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.utils.cache.file_cache import get_media_cache
from refiner.pipeline.utils.cache.lease_cache import LeaseCache


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


def test_hydrate_video_decodes_video(tmp_path: Path) -> None:
    video_path = tmp_path / "episode.mp4"
    _write_video(video_path)
    row = DictRow(
        {"video": VideoFile(str(video_path), from_timestamp_s=0.0, to_timestamp_s=0.3)}
    )

    hydrated = asyncio.run(hydrate_video("video")(row))
    video = hydrated["video"]
    assert isinstance(video, DecodedVideo)
    assert len(video.frames) == 4
    assert video.width == 16
    assert video.height == 16


def test_map_async_hydration_preserves_row_order(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for i in range(5):
        video_path = tmp_path / f"episode-{i}.mp4"
        _write_video(video_path)
        rows.append(
            {
                "id": i,
                "video": VideoFile(
                    str(video_path),
                    from_timestamp_s=0.0,
                    to_timestamp_s=0.3,
                ),
            }
        )

    out = (
        mdr.from_items(rows)
        .map_async(hydrate_video("video"), max_in_flight=2)
        .materialize()
    )

    assert [int(r["id"]) for r in out] == [0, 1, 2, 3, 4]
    assert all(len(row["video"].frames) == 4 for row in out)


def test_hydrate_video_can_decode_multiple_columns(tmp_path: Path) -> None:
    video_a = tmp_path / "episode-a.mp4"
    video_b = tmp_path / "episode-b.mp4"
    _write_video(video_a)
    _write_video(video_b)
    row = DictRow(
        {
            "left": VideoFile(str(video_a), from_timestamp_s=0.0, to_timestamp_s=0.3),
            "right": VideoFile(str(video_b), from_timestamp_s=0.0, to_timestamp_s=0.3),
        }
    )

    hydrated = asyncio.run(hydrate_video("left", "right")(row))

    assert isinstance(hydrated["left"], DecodedVideo)
    assert isinstance(hydrated["right"], DecodedVideo)
    assert len(hydrated["left"].frames) == 4
    assert len(hydrated["right"].frames) == 4


def test_media_cache_async_context_reuses_download() -> None:
    uri = "memory://context-manager.bin"
    with VideoFile(uri).open("wb") as f:
        f.write(b"context-manager")

    cache = get_media_cache("context-manager")

    async def exercise() -> tuple[str, str]:
        data_file = DataFile.resolve(uri)
        async with cache.cached(file=data_file) as first_path:
            assert os.path.exists(first_path)
            async with cache.cached(file=data_file) as second_path:
                assert os.path.exists(second_path)
                return first_path, second_path

    path, second_path = asyncio.run(exercise())
    assert path == second_path


def test_media_cache_file_lease_can_be_acquired_and_released_twice() -> None:
    uri = "memory://lease-same-file.bin"
    with VideoFile(uri).open("wb") as f:
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


def test_lease_cache_waiters_share_one_inflight_load() -> None:
    class _TestLeaseCache(LeaseCache[str, str]):
        def __init__(self) -> None:
            super().__init__(max_entries=4)
            self.create_calls = 0
            self.started = asyncio.Event()
            self.unblock = asyncio.Event()

        async def _create_resource(self, key: str) -> tuple[str, int]:
            self.create_calls += 1
            self.started.set()
            await self.unblock.wait()
            return f"resource:{key}", 1

        def _close_resource(self, resource: str) -> None:
            return

    async def exercise() -> None:
        cache = _TestLeaseCache()

        first = asyncio.create_task(cache.acquire("same"))
        await cache.started.wait()
        second = asyncio.create_task(cache.acquire("same"))

        cache.unblock.set()
        lease1, lease2 = await asyncio.gather(first, second)
        try:
            assert lease1.resource == "resource:same"
            assert lease2.resource == "resource:same"
            assert cache.create_calls == 1
        finally:
            lease1.release()
            lease2.release()

    asyncio.run(exercise())
