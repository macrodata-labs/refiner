from __future__ import annotations

from collections import deque
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from typing import Any

from refiner.pipeline.utils.cache.decoder_cache import get_opened_video_source_cache
from refiner.video.types import VideoFile

_FRAME_TIMESTAMP_EPSILON_S = 1e-6


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    index: int
    pts: int | None
    timestamp_s: float | None
    width: int
    height: int
    frame: Any


@dataclass(frozen=True, slots=True)
class DecodedFrameWindow:
    anchor: DecodedVideoFrame
    offsets: tuple[int, ...]
    frames: tuple[DecodedVideoFrame | None, ...]


def _frame_timestamp_s(frame: Any) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


def _seek_stream(
    *,
    container: Any,
    stream: Any,
    clip_from: float,
) -> None:
    if stream.time_base is None:
        return
    seek_ts = int(clip_from / float(stream.time_base))
    try:
        container.seek(seek_ts, stream=stream)
    except Exception:
        try:
            container.seek(max(0, seek_ts), any_frame=True, stream=stream)
        except Exception:
            pass


def _iter_selected_frames(
    *,
    container: Any,
    stream: Any,
    clip_from: float,
    clip_to: float | None,
    seek: bool,
) -> Iterator[Any]:
    if seek:
        _seek_stream(container=container, stream=stream, clip_from=clip_from)

    for frame in container.decode(stream):
        timestamp_s = _frame_timestamp_s(frame)
        if timestamp_s is None:
            continue
        if timestamp_s + _FRAME_TIMESTAMP_EPSILON_S < clip_from:
            continue
        if clip_to is not None and timestamp_s - _FRAME_TIMESTAMP_EPSILON_S >= clip_to:
            break
        yield frame


async def iter_frames(
    video: VideoFile,
    *,
    cache_key: str = "default",
) -> AsyncGenerator[DecodedVideoFrame, None]:
    lease = await get_opened_video_source_cache(name=cache_key).acquire(video.uri)
    source = lease.resource
    clip_from = float(video.from_timestamp_s or 0.0)
    clip_to = video.to_timestamp_s

    try:
        for index, frame in enumerate(
            _iter_selected_frames(
                container=source.container,
                stream=source.stream,
                clip_from=clip_from,
                clip_to=clip_to,
                seek=True,
            )
        ):
            yield DecodedVideoFrame(
                index=index,
                pts=frame.pts,
                timestamp_s=_frame_timestamp_s(frame),
                width=int(frame.width),
                height=int(frame.height),
                frame=frame,
            )
    finally:
        lease.release()


async def iter_frame_windows(
    video: VideoFile,
    *,
    offsets: list[int] | tuple[int, ...],
    stride: int = 1,
    drop_incomplete: bool = True,
    cache_key: str = "default",
) -> AsyncGenerator[DecodedFrameWindow, None]:
    offsets_tuple = tuple(int(offset) for offset in offsets)
    if not offsets_tuple:
        raise ValueError("offsets must contain at least one entry")
    if 0 not in offsets_tuple:
        raise ValueError("offsets must include 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    min_offset = min(offsets_tuple)
    max_offset = max(offsets_tuple)
    buffer: deque[DecodedVideoFrame] = deque()
    buffer_by_index: dict[int, DecodedVideoFrame] = {}
    next_anchor_index = 0
    next_pending_anchor_index = 0

    def _trim_buffer(current_index: int) -> None:
        min_needed_index = min(
            next_anchor_index + min_offset, next_pending_anchor_index + min_offset
        )
        if max_offset > 0:
            min_needed_index = min(min_needed_index, current_index - max_offset)
        while buffer and buffer[0].index < min_needed_index:
            dropped = buffer.popleft()
            buffer_by_index.pop(dropped.index, None)

    def _build_window(anchor: DecodedVideoFrame) -> DecodedFrameWindow | None:
        frames: list[DecodedVideoFrame | None] = []

        for offset in offsets_tuple:
            target_index = anchor.index + offset
            if offset == 0:
                frames.append(anchor)
            else:
                frames.append(buffer_by_index.get(target_index))

        if drop_incomplete and any(frame is None for frame in frames):
            return None

        return DecodedFrameWindow(
            anchor=anchor,
            offsets=offsets_tuple,
            frames=tuple(frames),
        )

    async for frame in iter_frames(
        video,
        cache_key=cache_key,
    ):
        buffer.append(frame)
        buffer_by_index[frame.index] = frame

        while next_anchor_index <= frame.index:
            next_anchor_index += stride

        while next_pending_anchor_index <= frame.index:
            anchor = buffer_by_index.get(next_pending_anchor_index)
            if anchor is None:
                next_pending_anchor_index += stride
                continue
            if anchor.index + max_offset > frame.index:
                break
            window = _build_window(anchor)
            if window is not None:
                yield window
            next_pending_anchor_index += stride

        _trim_buffer(frame.index)

    while next_pending_anchor_index < next_anchor_index:
        anchor = buffer_by_index.get(next_pending_anchor_index)
        next_pending_anchor_index += stride
        if anchor is None:
            continue
        window = _build_window(anchor)
        if window is not None:
            yield window
        _trim_buffer(anchor.index + max_offset)


__all__ = [
    "DecodedFrameWindow",
    "DecodedVideoFrame",
    "iter_frame_windows",
    "iter_frames",
]
