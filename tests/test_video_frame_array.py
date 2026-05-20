from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from refiner.io import DataFolder
from refiner.video import VideoFrameArray, VideoStreamWriter, VideoTranscodeConfig


async def _collect_frame_shapes(video: VideoFrameArray) -> list[tuple[int, int]]:
    return [(frame.height, frame.width) async for frame in video.iter_frames()]


def test_video_frame_array_normalizes_frame_sequence() -> None:
    frames = [
        np.zeros((8, 10, 3), dtype=np.uint8),
        np.ones((8, 10, 3), dtype=np.uint8),
    ]

    video = VideoFrameArray(frames, fps=12)

    assert video.shape == (2, 8, 10, 3)
    assert video.frame_count == 2
    assert video.fps == 12
    assert len(list(video.iter_frame_arrays())) == 2


def test_video_frame_array_equality_is_identity() -> None:
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)

    first = VideoFrameArray(frames, fps=12)
    second = VideoFrameArray(frames.copy(), fps=12)

    assert first == first
    assert first != second


def test_video_frame_array_open_returns_encoded_video_bytes() -> None:
    pytest.importorskip("av")
    frames = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 200
    video = VideoFrameArray(frames, fps=10)

    with video.open() as stream:
        assert stream.read(8)

    assert asyncio.run(_collect_frame_shapes(video)) == [(8, 10), (8, 10), (8, 10)]


def test_video_stream_writer_accepts_video_frame_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("av")
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 180
    video = VideoFrameArray(frames, fps=10)
    monkeypatch.setattr(
        VideoFrameArray,
        "open",
        lambda *_args, **_kwargs: pytest.fail("writer should not encode/decode arrays"),
    )
    output = DataFolder.resolve(str(tmp_path / "out"))
    writer = VideoStreamWriter(
        folder=output,
        stream_key="front",
        transcode_config=VideoTranscodeConfig(codec="mpeg4", pix_fmt="yuv420p"),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/file-{file_index:03d}.mp4",
    )

    written = asyncio.run(writer.write_video(video, force_transcode=True))
    writer.close()

    assert written.mode == "transcode"
    assert written.segment.stream_key == "front"
    assert written.segment.fps == 10
    assert written.segment.width == 16
    assert written.segment.height == 16
    assert (tmp_path / "out" / "videos" / "front" / "file-000.mp4").exists()


def test_video_stream_writer_writes_all_video_frame_array_frames(
    tmp_path: Path,
) -> None:
    pytest.importorskip("av")
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    video = VideoFrameArray(frames, fps=10)
    output = DataFolder.resolve(str(tmp_path / "out"))
    writer = VideoStreamWriter(
        folder=output,
        stream_key="front",
        transcode_config=VideoTranscodeConfig(codec="mpeg4", pix_fmt="yuv420p"),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/file-{file_index:03d}.mp4",
    )

    written = asyncio.run(writer.write_video(video))
    writer.close()

    assert written.segment.from_timestamp == pytest.approx(0.0)
    assert written.segment.to_timestamp == pytest.approx(0.4)
