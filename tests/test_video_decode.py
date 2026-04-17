from __future__ import annotations

import asyncio

import numpy as np

import refiner as mdr
from refiner.io import DataFile


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


async def _collect_frames(video: mdr.video.VideoFile):
    return [frame async for frame in video.iter_frames()]


async def _collect_windows(
    video: mdr.video.VideoFile,
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
