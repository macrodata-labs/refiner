from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
from pathlib import Path
import time
from typing import Iterator

import av
import numpy as np
import refiner as mdr
import pytest

from refiner.io import DataFile
from refiner.media import (
    DecodedVideo,
    MediaFile,
    Video,
    get_media_cache,
    reset_video_decoder_cache,
    hydrate_media,
    reset_media_cache,
)
from refiner.media.video.utils import decode_video_segment_frames
from refiner.sources.row import DictRow


@pytest.fixture(autouse=True)
def _reset_media_caches() -> Iterator[None]:
    reset_media_cache()
    reset_video_decoder_cache()
    yield
    reset_media_cache()
    reset_video_decoder_cache()


def test_hydrate_media_video_decode_false_rejects_timestamps() -> None:
    row = DictRow(
        {
            "video": Video(
                media=MediaFile("memory://video.mp4"),
                video_key="cam",
                from_timestamp_s=0.0,
                to_timestamp_s=1.0,
            )
        }
    )

    with pytest.raises(ValueError, match="decode=False"):
        asyncio.run(hydrate_media("video", decode=False)(row))


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
    assert isinstance(video.media, MediaFile)
    assert video.media.bytes_cache == payload


def test_hydrate_media_video_decode_true_clips_timestamped_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for idx in range(10):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 20
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    row = DictRow(
        {
            "video": Video(
                media=MediaFile(str(source)),
                video_key="cam",
                from_timestamp_s=0.2,
                to_timestamp_s=0.6,
            )
        }
    )

    hydrated = asyncio.run(hydrate_media("video", decode=True)(row))
    video = hydrated["video"]
    assert isinstance(video, Video)
    assert isinstance(video.media, DecodedVideo)
    video = video.media
    assert video.frame_count > 0
    assert video.frame_count < 10
    assert video.width == 16
    assert video.height == 16
    assert video.pix_fmt == "rgb24"
    first = video.frames[0]
    assert isinstance(first, np.ndarray)
    assert video.uri == f"{source}"


def test_hydrate_media_video_decode_true_fast_ffmpeg_backend(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for decode_backend='ffmpeg'")

    source = tmp_path / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for idx in range(10):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 20
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    row = DictRow(
        {
            "video": Video(
                media=MediaFile(str(source)),
                video_key="cam",
                from_timestamp_s=0.2,
                to_timestamp_s=0.6,
            )
        }
    )

    hydrated = asyncio.run(
        hydrate_media("video", decode=True, decode_backend="ffmpeg")(row)
    )
    video = hydrated["video"]
    assert isinstance(video, Video)
    assert isinstance(video.media, DecodedVideo)
    assert video.media.frame_count > 0


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


def test_media_file_cleanup_clears_hydrated_state() -> None:
    handle = MediaFile("memory://video.bin")
    with handle.open("wb") as f:
        f.write(b"0123456789")

    with handle.cached_path(suffix=".bin") as local_path:
        assert os.path.exists(local_path)
        assert Path(local_path).read_bytes() == b"0123456789"
    assert os.path.exists(local_path)
    assert handle.cache_bytes() == b"0123456789"

    handle.cleanup()
    assert handle.local_path is None
    assert handle.bytes_cache is None
    assert os.path.exists(local_path)


def test_media_cache_dedupes_parallel_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        fut_a = pool.submit(
            lambda: _cached_path_once(a, suffix=".bin", cache_name="parallel")
        )
        fut_b = pool.submit(
            lambda: _cached_path_once(b, suffix=".bin", cache_name="parallel")
        )
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
        with cache.cached(
            file=data_file,
        ) as path:
            assert os.path.exists(path)
            assert Path(path).read_bytes() == b"context-manager"

        with cache.cached(
            file=data_file,
        ) as second_path:
            assert os.path.exists(second_path)
            assert second_path == path

    assert calls["count"] == 1


def test_named_media_caches_are_isolated() -> None:
    uri = "memory://isolated-cache.bin"
    seed = MediaFile(uri)
    with seed.open("wb") as f:
        f.write(b"isolated")

    a = MediaFile(uri)
    b = MediaFile(uri)
    path_a = _cached_path_once(a, suffix=".bin", cache_name="cache-a")
    path_b = _cached_path_once(b, suffix=".bin", cache_name="cache-b")

    assert path_a != path_b
    assert os.path.exists(path_a)
    assert os.path.exists(path_b)


def test_hydrate_media_defaults_to_column_scoped_cache() -> None:
    uri = "memory://column-scoped.bin"
    with MediaFile(uri).open("wb") as f:
        f.write(b"column-cache")

    row = DictRow({"cam_left": uri, "cam_right": uri})
    left_row = asyncio.run(hydrate_media("cam_left")(row))
    right_row = asyncio.run(hydrate_media("cam_right")(row))

    left_media = left_row["cam_left"]
    right_media = right_row["cam_right"]
    assert isinstance(left_media, MediaFile)
    assert isinstance(right_media, MediaFile)
    assert left_media.bytes_cache == b"column-cache"
    assert right_media.bytes_cache == b"column-cache"


def test_media_cache_evicts_oldest_entry_at_capacity() -> None:
    get_media_cache("evict-test", max_entries=1, max_bytes=1024 * 1024)

    uri_a = "memory://evict-a.bin"
    uri_b = "memory://evict-b.bin"
    with MediaFile(uri_a).open("wb") as f:
        f.write(b"A")
    with MediaFile(uri_b).open("wb") as f:
        f.write(b"B")

    first = MediaFile(uri_a)
    second = MediaFile(uri_b)
    first_path = _cached_path_once(first, suffix=".bin", cache_name="evict-test")
    second_path = _cached_path_once(second, suffix=".bin", cache_name="evict-test")

    assert os.path.exists(second_path)
    assert not os.path.exists(first_path)


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for idx in range(10):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[..., 0] = idx * 20
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _cached_path_once(
    media: MediaFile,
    *,
    suffix: str | None = None,
    cache_name: str = "default",
) -> str:
    with media.cached_path(suffix=suffix, cache_name=cache_name) as path:
        return path


def test_decode_video_segment_frames_uses_decoder_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    _write_test_video(source)

    call_count = {"open": 0}
    original_open = av.open

    def counted_open(*args, **kwargs):
        call_count["open"] += 1
        return original_open(*args, **kwargs)

    import refiner.media.video.utils as video_utils

    monkeypatch.setattr(video_utils.av, "open", counted_open)

    # First decode warms cache.
    frames1 = decode_video_segment_frames(
        local_path=str(source),
        from_timestamp_s=0.0,
        to_timestamp_s=0.5,
        decoder_cache_name="test-reuse",
    )[0]
    # Second decode with same clip should reuse decoder without opening again.
    frames2 = decode_video_segment_frames(
        local_path=str(source),
        from_timestamp_s=0.0,
        to_timestamp_s=0.5,
        decoder_cache_name="test-reuse",
    )[0]

    assert call_count["open"] == 1
    assert isinstance(frames1, tuple)
    assert frames1 == frames2


def test_decode_video_segment_frames_concurrent_reuses_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    _write_test_video(source)

    call_count = {"open": 0}
    original_open = av.open

    def counted_open(*args, **kwargs):
        call_count["open"] += 1
        return original_open(*args, **kwargs)

    import refiner.media.video.utils as video_utils

    monkeypatch.setattr(video_utils.av, "open", counted_open)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(
            decode_video_segment_frames,
            local_path=str(source),
            from_timestamp_s=0.0,
            to_timestamp_s=0.5,
            decoder_cache_name="test-reuse",
        )
        fut_b = pool.submit(
            decode_video_segment_frames,
            local_path=str(source),
            from_timestamp_s=0.0,
            to_timestamp_s=0.5,
            decoder_cache_name="test-reuse",
        )
        first = fut_a.result()
        second = fut_b.result()

    assert call_count["open"] == 1
    assert first[0] == second[0]
