from __future__ import annotations

import io
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refiner.video.remux import (
    RemuxWriter,
    prepared_source_is_remuxable,
    prepare_video_source,
    video_from_timestamp_s,
    video_to_timestamp_s,
)

if TYPE_CHECKING:
    import av

    from refiner.video.types import VideoFile
    from refiner.video.transcode import VideoTranscodeConfig

_FRAME_TIMESTAMP_EPSILON_S = 1e-6


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    index: int
    pts: int | None
    timestamp_s: float | None
    width: int
    height: int
    frame: av.VideoFrame


@dataclass(frozen=True, slots=True)
class DecodedFrameWindow:
    anchor: DecodedVideoFrame
    offsets: tuple[int, ...]
    frames: tuple[DecodedVideoFrame | None, ...]


class _NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        pass


async def export_clip(
    video: VideoFile,
    *,
    force_transcode: bool = False,
    transcode_config: VideoTranscodeConfig | None = None,
) -> bytes:
    from refiner.video.transcode import TranscodeWriter, VideoTranscodeConfig

    config = transcode_config or VideoTranscodeConfig()
    prepared = await prepare_video_source(video=video)
    output_file = _NonClosingBytesIO()
    try:
        if not force_transcode and prepared_source_is_remuxable(prepared):
            probe = prepared.probe
            if probe is None:
                raise RuntimeError("Remux path selected without a source probe")
            writer = RemuxWriter.open_file(
                output_file=output_file,
                probe=probe,
                movflags=None,
            )
            writer.append_prepared_video(prepared)
        else:
            fps = (
                int(prepared.probe.fps)
                if prepared.probe is not None and prepared.probe.fps is not None
                else None
            )
            if fps is None:
                raise ValueError("Prepared transcode item is missing FPS")
            writer = TranscodeWriter.open_file(
                output_file=output_file,
                config=config,
                fps=fps,
                movflags=None,
            )
            writer.append_prepared_video(
                prepared_source=prepared,
            )
        writer.close()
        return output_file.getvalue()
    finally:
        prepared.close()


async def iter_frames(
    video: VideoFile,
) -> AsyncIterator[DecodedVideoFrame]:
    prepared = await prepare_video_source(video=video)
    try:
        frames = _iter_selected_frames(
            container=prepared.container,
            stream=prepared.stream,
            clip_from=video_from_timestamp_s(prepared.video),
            clip_to=video_to_timestamp_s(prepared.video),
            seek=True,
        )
        for index, frame in enumerate(frames):
            yield DecodedVideoFrame(
                index=index,
                pts=None if frame.pts is None else int(frame.pts),
                timestamp_s=_frame_timestamp_s(frame),
                width=int(frame.width),
                height=int(frame.height),
                frame=frame,
            )
    finally:
        prepared.close()


async def iter_frame_windows(
    video: VideoFile,
    *,
    offsets: Sequence[int],
    stride: int = 1,
    drop_incomplete: bool = True,
) -> AsyncIterator[DecodedFrameWindow]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    normalized_offsets = tuple(int(offset) for offset in offsets)
    min_offset = min(normalized_offsets)
    max_offset = max(normalized_offsets)
    buffer: deque[DecodedVideoFrame] = deque()
    frames_by_index: dict[int, DecodedVideoFrame] = {}
    pending_anchor_indexes: deque[int] = deque()

    async for frame in iter_frames(video):
        buffer.append(frame)
        frames_by_index[frame.index] = frame
        if frame.index % stride == 0:
            pending_anchor_indexes.append(frame.index)

        while (
            pending_anchor_indexes
            and frame.index >= pending_anchor_indexes[0] + max_offset
        ):
            anchor_index = pending_anchor_indexes.popleft()
            window = _build_frame_window(
                anchor_index=anchor_index,
                offsets=normalized_offsets,
                frames_by_index=frames_by_index,
                drop_incomplete=drop_incomplete,
            )
            if window is not None:
                yield window

        keep_from = (
            pending_anchor_indexes[0] + min(0, min_offset)
            if pending_anchor_indexes
            else _next_anchor_index(frame.index, stride) + min(0, min_offset)
        )
        while buffer and buffer[0].index < keep_from:
            dropped = buffer.popleft()
            frames_by_index.pop(dropped.index, None)

    while pending_anchor_indexes:
        anchor_index = pending_anchor_indexes.popleft()
        window = _build_frame_window(
            anchor_index=anchor_index,
            offsets=normalized_offsets,
            frames_by_index=frames_by_index,
            drop_incomplete=drop_incomplete,
        )
        if window is not None:
            yield window


def _build_frame_window(
    *,
    anchor_index: int,
    offsets: tuple[int, ...],
    frames_by_index: dict[int, DecodedVideoFrame],
    drop_incomplete: bool,
) -> DecodedFrameWindow | None:
    anchor = frames_by_index.get(anchor_index)
    if anchor is None:
        return None

    frames: list[DecodedVideoFrame | None] = []
    for offset in offsets:
        frame = frames_by_index.get(anchor_index + offset)
        if frame is None and drop_incomplete:
            return None
        frames.append(frame)

    return DecodedFrameWindow(
        anchor=anchor,
        offsets=offsets,
        frames=tuple(frames),
    )


def _next_anchor_index(frame_index: int, stride: int) -> int:
    return ((frame_index // stride) + 1) * stride


def _iter_selected_frames(
    *,
    container: Any,
    stream: Any,
    clip_from: float,
    clip_to: float | None,
    seek: bool,
):
    if seek and stream.time_base is not None:
        seek_ts = int(clip_from / float(stream.time_base))
        try:
            container.seek(seek_ts, stream=stream)
        except Exception:
            try:
                container.seek(max(0, seek_ts), any_frame=True, stream=stream)
            except Exception:
                pass

    for frame in container.decode(stream):
        ts = _frame_timestamp_s(frame)
        if ts is None:
            continue
        if ts + _FRAME_TIMESTAMP_EPSILON_S < clip_from:
            continue
        if clip_to is not None and ts - _FRAME_TIMESTAMP_EPSILON_S >= clip_to:
            break
        yield frame


def _frame_timestamp_s(frame: Any) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


__all__ = [
    "DecodedFrameWindow",
    "DecodedVideoFrame",
    "export_clip",
    "iter_frame_windows",
    "iter_frames",
]
