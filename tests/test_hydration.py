from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import refiner as mdr

from refiner.io import DataFile
from refiner.media import MediaFile, Video, get_media_cache, hydrate_media
from refiner.pipeline.data.row import DictRow


def test_video_decode_true_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        Video(media=MediaFile("memory://video.mp4"), video_key="cam", decode=True)


def test_hydrate_media_turns_string_into_media_file(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"abc123"
    payload_path.write_bytes(payload)

    row = DictRow({"blob_uri": str(payload_path)})
    hydrated = asyncio.run(hydrate_media("blob_uri")(row))
    value = hydrated["blob_uri"]

    assert isinstance(value, MediaFile)
    assert value.bytes_cache == payload


def test_hydrate_media_hydrates_video_wrapper_bytes(tmp_path: Path) -> None:
    payload_path = tmp_path / "blob.bin"
    payload = b"bytes-mode"
    payload_path.write_bytes(payload)
    row = DictRow({"video": Video(media=MediaFile(str(payload_path)), video_key="cam")})

    hydrated = asyncio.run(hydrate_media("video")(row))
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
        .map_async(hydrate_media("blob_uri"), max_in_flight=2)
        .materialize()
    )

    assert [int(r["id"]) for r in out] == [0, 1, 2, 3, 4]
    for row in out:
        idx = int(row["id"])
        media = row["blob_uri"]
        assert isinstance(media, MediaFile)
        assert media.bytes_cache == payloads[idx]


def _cached_path_once(handle: MediaFile, *, cache_name: str) -> str:
    return asyncio.run(handle.cache_file(cache_name=cache_name))


def test_media_cache_dedupes_parallel_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    uri = "memory://shared-video.bin"
    seed = MediaFile(uri)
    with seed.open("wb") as f:
        f.write(b"shared")

    calls = {"count": 0}
    import refiner.media.cache as media_cache

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

    import refiner.media.cache as media_cache

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


def test_media_cache_lease_can_be_acquired_and_released_without_cached() -> None:
    cache = get_media_cache("lease-without-cache")

    async def exercise() -> None:
        await cache.get_lease()
        cache.release_lease()

    asyncio.run(exercise())


def test_media_cache_lease_blocks_until_release() -> None:
    cache = get_media_cache("lease-blocking", max_entries=1)
    events: list[str] = []
    unblock = asyncio.Event()

    async def holder() -> None:
        await cache.get_lease()
        events.append("first-held")
        await unblock.wait()
        events.append("first-release")
        cache.release_lease()

    async def waiter() -> None:
        await asyncio.sleep(0.05)
        await cache.get_lease()
        events.append("second-held")
        cache.release_lease()

    async def run() -> None:
        holder_task = asyncio.create_task(holder())
        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.1)
        assert events == ["first-held"]
        unblock.set()
        await asyncio.gather(holder_task, waiter_task)
        assert events == ["first-held", "first-release", "second-held"]

    asyncio.run(run())


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
