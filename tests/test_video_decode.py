from __future__ import annotations

import asyncio

import numpy as np
import pytest

import refiner as mdr
from refiner.io import DataFile
from refiner.io import DataFolder
from refiner.video import VideoStreamWriter, VideoTranscodeConfig


def _write_video(path, *, num_frames: int = 5, fps: int = 5) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "yuv420p"

        for value in range(num_frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((4, 4, 3), value, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


async def _collect_frames(video: mdr.video.VideoSource):
    return [frame async for frame in video.iter_frames()]


async def _collect_windows(
    video: mdr.video.VideoSource,
    *,
    offsets: list[int],
    stride: int = 1,
    drop_incomplete: bool = True,
):
    return [
        window
        async for window in video.iter_frame_windows(
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )
    ]


def test_iter_frames_respects_clip_bounds(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=5, fps=5)
    video = mdr.video.VideoFile(
        DataFile.resolve(path),
        from_timestamp_s=0.2,
        to_timestamp_s=0.7,
    )

    frames = asyncio.run(_collect_frames(video))

    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_s for frame in frames] == [0.2, 0.4, 0.6]


def test_video_frame_array_clip_returns_frame_view() -> None:
    frames = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in range(6)])
    video = mdr.video.VideoFrameArray(frames, fps=10)

    clipped = video.clipped(from_timestamp_s=0.2, to_timestamp_s=0.5)

    assert isinstance(clipped, mdr.video.VideoFrameArray)
    clipped_frames = list(clipped.iter_frame_arrays())
    assert len(clipped_frames) == 3
    assert clipped_frames[0].shape == (4, 4, 3)
    assert [int(frame[0, 0, 0]) for frame in clipped_frames] == [2, 3, 4]


def test_video_frame_array_iter_frames() -> None:
    frames = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3)])
    video = mdr.video.VideoFrameArray(frames, fps=5)

    decoded = asyncio.run(_collect_frames(video))
    arrays = list(video.iter_frame_arrays())

    assert [frame.index for frame in decoded] == [0, 1, 2]
    assert [frame.timestamp_s for frame in decoded] == [0.0, 0.2, 0.4]
    assert [int(frame[0, 0, 0]) for frame in arrays] == [0, 1, 2]


def test_video_frame_sequence_iterates_without_stacking() -> None:
    calls = 0

    def frames():
        nonlocal calls
        for value in range(3):
            calls += 1
            yield np.full((4, 4, 3), value, dtype=np.uint8)

    video = mdr.video.VideoFrameSequence(frames, fps=5, frame_count=3)

    decoded = asyncio.run(_collect_frames(video))
    arrays = list(video.iter_frame_arrays())

    assert video.frame_count == 3
    assert [frame.index for frame in decoded] == [0, 1, 2]
    assert [frame.timestamp_s for frame in decoded] == [0.0, 0.2, 0.4]
    assert [int(frame[0, 0, 0]) for frame in arrays] == [0, 1, 2]
    assert calls == 6


def test_video_frame_sequence_rejects_one_shot_iterators() -> None:
    frames = (np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3))

    with pytest.raises(ValueError, match="frames must be repeatable"):
        mdr.video.VideoFrameSequence(frames, fps=5, frame_count=3)


def test_video_frame_sequence_clips_frame_count() -> None:
    video = mdr.video.VideoFrameSequence(
        lambda: (np.full((4, 4, 3), value, dtype=np.uint8) for value in range(5)),
        fps=2,
        frame_count=5,
    )

    clipped = video.clipped(from_timestamp_s=0.5, to_timestamp_s=1.5)
    decoded = asyncio.run(_collect_frames(clipped))

    assert clipped.frame_count == 2
    assert [int(frame[0, 0, 0]) for frame in clipped.iter_frame_arrays()] == [1, 2]
    assert [frame.index for frame in decoded] == [0, 1]
    assert [frame.timestamp_s for frame in decoded] == [0.0, 0.5]


def test_video_frame_array_preserves_fractional_fps() -> None:
    frames = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3)])
    video = mdr.video.VideoFrameArray(frames, fps=29.97)

    decoded = asyncio.run(_collect_frames(video))

    assert video.fps == 29.97
    assert decoded[1].timestamp_s == pytest.approx(1 / 29.97)


def test_video_stream_writer_accepts_video_frame_array(tmp_path) -> None:
    video = mdr.video.VideoFrameArray(
        np.stack([np.full((8, 8, 3), value, dtype=np.uint8) for value in range(4)]),
        fps=29.97,
    )
    writer = VideoStreamWriter(
        folder=DataFolder.resolve(tmp_path),
        stream_key="camera",
        transcode_config=VideoTranscodeConfig(),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/file-{file_index:03d}.mp4",
    )

    written = asyncio.run(video.write_to(writer))
    writer.close()

    assert written.segment.fps == 29.97
    assert written.segment.to_timestamp == pytest.approx(4 / 29.97)


def test_video_stream_writer_accepts_video_frame_sequence(tmp_path) -> None:
    video = mdr.video.VideoFrameSequence(
        lambda: (np.full((8, 8, 3), value, dtype=np.uint8) for value in range(4)),
        fps=29.97,
        frame_count=4,
    )
    writer = VideoStreamWriter(
        folder=DataFolder.resolve(tmp_path),
        stream_key="camera",
        transcode_config=VideoTranscodeConfig(),
        video_bytes_limit=1024 * 1024,
        output_rel_template="videos/{stream_key}/file-{file_index:03d}.mp4",
    )

    written = asyncio.run(video.write_to(writer))
    writer.close()

    assert written.segment.fps == 29.97
    assert written.segment.to_timestamp == pytest.approx(4 / 29.97)


def test_iter_frame_windows_supports_lookahead(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=5, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    windows = asyncio.run(
        _collect_windows(
            video,
            offsets=[-1, 0, 1],
            stride=2,
            drop_incomplete=False,
        )
    )

    assert [window.anchor.index for window in windows] == [0, 2, 4]
    assert [
        [frame.index if frame is not None else None for frame in window.frames]
        for window in windows
    ] == [
        [None, 0, 1],
        [1, 2, 3],
        [3, 4, None],
    ]


def test_iter_frame_windows_can_drop_incomplete_windows(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=5, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    windows = asyncio.run(
        _collect_windows(
            video,
            offsets=[-1, 0, 1],
            stride=2,
            drop_incomplete=True,
        )
    )

    assert [window.anchor.index for window in windows] == [2]


def test_iter_frame_windows_supports_positive_offsets_only(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=5, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    windows = asyncio.run(
        _collect_windows(
            video,
            offsets=[1],
            stride=2,
            drop_incomplete=False,
        )
    )

    assert [window.anchor.index for window in windows] == [0, 2, 4]
    assert [
        [frame.index if frame is not None else None for frame in window.frames]
        for window in windows
    ] == [
        [1],
        [3],
        [None],
    ]
