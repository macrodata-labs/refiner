from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

import refiner as mdr
from refiner.io import DataFile
from refiner.pipeline.utils.cache.decoder_cache import get_opened_video_source_cache
from refiner.video.remux import reset_opened_video_source_cache

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2: Any = None


@pytest.fixture(autouse=True)
def _reset_opened_source_cache() -> Iterator[None]:
    reset_opened_video_source_cache()
    yield
    reset_opened_video_source_cache()


def _write_video(path: Path, *, fps: int = 10, frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"

        for idx in range(frames):
            image = np.full((16, 16, 3), idx * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


def test_iter_frames_decodes_videofile_bounds(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.2,
        to_timestamp_s=0.5,
    )

    async def _collect() -> list[mdr.video.DecodedVideoFrame]:
        return [frame async for frame in mdr.video.iter_frames(video)]

    frames = asyncio.run(_collect())

    assert len(frames) == 4
    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_s for frame in frames] == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )
    assert all(frame.width == 16 for frame in frames)
    assert all(frame.height == 16 for frame in frames)


def test_iter_frames_releases_cache_lease_on_close(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.0,
        to_timestamp_s=0.6,
    )

    async def _consume_one_and_reacquire() -> None:
        frames = mdr.video.iter_frames(video, cache_key="iter-close")
        first = await anext(frames)
        assert first.index == 0
        await frames.aclose()

        cache = get_opened_video_source_cache(name="iter-close")
        lease = await asyncio.wait_for(cache.acquire(video.uri), timeout=1.0)
        lease.release()

    asyncio.run(_consume_one_and_reacquire())


def test_iter_frame_windows_supports_stride_and_history(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=8)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.0,
        to_timestamp_s=0.7,
    )

    async def _collect() -> list[mdr.video.DecodedFrameWindow]:
        return [
            window
            async for window in mdr.video.iter_frame_windows(
                video,
                offsets=[-2, 0],
                stride=2,
            )
        ]

    windows = asyncio.run(_collect())

    assert [window.anchor.index for window in windows] == [2, 4, 6]
    assert [
        [frame.index if frame is not None else None for frame in window.frames]
        for window in windows
    ] == [[0, 2], [2, 4], [4, 6]]


def test_iter_frame_windows_supports_lookahead_and_partial_tail(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.0,
        to_timestamp_s=0.5,
    )

    async def _collect() -> list[mdr.video.DecodedFrameWindow]:
        return [
            window
            async for window in mdr.video.iter_frame_windows(
                video,
                offsets=[0, 2],
                stride=2,
                drop_incomplete=False,
            )
        ]

    windows = asyncio.run(_collect())

    assert [window.anchor.index for window in windows] == [0, 2, 4]
    assert [
        [frame.index if frame is not None else None for frame in window.frames]
        for window in windows
    ] == [[0, 2], [2, 4], [4, None]]


def test_export_clip_bytes_uses_video_writer_path(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.2,
        to_timestamp_s=0.5,
    )

    payload = asyncio.run(mdr.video.export_clip_bytes(video))

    assert payload

    with av.open(io.BytesIO(payload), mode="r") as container:
        stream = next(item for item in container.streams if item.type == "video")
        timestamps = [
            float(frame.pts * frame.time_base)
            for frame in container.decode(stream)
            if isinstance(frame, av.VideoFrame)
            and frame.pts is not None
            and frame.time_base is not None
        ]

    assert timestamps == pytest.approx([0.0, 0.1, 0.2])


@pytest.mark.skipif(cv2 is None, reason="opencv-python is not installed")
def test_export_clip_bytes_is_readable_by_opencv(tmp_path: Path) -> None:
    src_video = tmp_path / "source" / "episode.mp4"
    _write_video(src_video, fps=10, frames=6)

    video = mdr.video.VideoFile(
        DataFile.resolve(str(src_video)),
        from_timestamp_s=0.0,
        to_timestamp_s=0.5,
    )

    payload = asyncio.run(mdr.video.export_clip_bytes(video))
    exported = tmp_path / "exported.mp4"
    exported.write_bytes(payload)

    assert cv2 is not None
    cap = cv2.VideoCapture(str(exported))
    assert cap.isOpened()
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 5
        assert float(cap.get(cv2.CAP_PROP_FPS)) == pytest.approx(10.0)
    finally:
        cap.release()
