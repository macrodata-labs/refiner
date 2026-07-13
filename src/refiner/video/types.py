from __future__ import annotations

import io
import math
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from fractions import Fraction
from typing import IO, TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from refiner.io import DataFile

if TYPE_CHECKING:
    from refiner.video.decode import DecodedFrameWindow, DecodedVideoFrame
    from refiner.video.transcode import FrameObserver
    from refiner.video.writer import VideoStreamWriter, WrittenVideo


_REMOTE_VIDEO_BLOCK_SIZE = 5 * 1024 * 1024


def _uses_s3(filesystem: Any) -> bool:
    protocol = getattr(filesystem, "protocol", None)
    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol or ())
    return any(str(item).lower() in {"s3", "s3a"} for item in protocols)


@runtime_checkable
class VideoSource(Protocol):
    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoSource": ...

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]: ...

    def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]: ...

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]: ...

    async def write_to(
        self,
        writer: VideoStreamWriter,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo: ...


def _compose_clip_bounds(
    *,
    current_from: float | None,
    current_to: float | None,
    from_timestamp_s: float | None,
    to_timestamp_s: float | None,
) -> tuple[float | None, float | None]:
    relative_from = 0.0 if from_timestamp_s is None else float(from_timestamp_s)
    if relative_from < 0:
        raise ValueError("from_timestamp_s must be >= 0")
    if to_timestamp_s is not None and float(to_timestamp_s) < relative_from:
        raise ValueError("to_timestamp_s must be >= from_timestamp_s")

    base_from = float(current_from or 0.0)
    next_from = base_from + relative_from
    next_to = (
        current_to if to_timestamp_s is None else base_from + float(to_timestamp_s)
    )
    if current_to is not None:
        current_to_value = float(current_to)
        if next_from > current_to_value:
            raise ValueError("clip start is beyond the current video view")
        if next_to is not None and next_to > current_to_value:
            raise ValueError("clip end is beyond the current video view")
    return next_from, next_to


@dataclass(frozen=True, slots=True)
class VideoFile:
    data_file: DataFile
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None

    @property
    def uri(self) -> str:
        return str(self.data_file)

    def open(self) -> IO[bytes]:
        # MP4 demuxing repeatedly alternates between small reads in the sample
        # tables near the start of the file and reads in the current media
        # region. s3fs's default ReadAheadCache retains only one region, so
        # PyAV can force a large S3 range refill for every tiny alternating
        # read. BlockCache retains both regions and avoids that cache thrash.
        if _uses_s3(self.data_file.fs):
            return self.data_file.open(
                mode="rb",
                cache_type="blockcache",
                block_size=_REMOTE_VIDEO_BLOCK_SIZE,
            )
        return self.data_file.open(mode="rb")

    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoFile":
        next_from, next_to = _compose_clip_bounds(
            current_from=self.from_timestamp_s,
            current_to=self.to_timestamp_s,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )
        return VideoFile(
            data_file=self.data_file,
            from_timestamp_s=next_from,
            to_timestamp_s=next_to,
        )

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        from refiner.video.decode import iter_encoded_frames

        return iter_encoded_frames(self)

    async def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]:
        async for frame in self.iter_frames():
            yield frame.frame.to_ndarray(format="rgb24")

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]:
        from refiner.video.decode import iter_frame_windows

        return iter_frame_windows(
            self,
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )

    async def write_to(
        self,
        writer: VideoStreamWriter,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo:
        return await writer.write_encoded_video(
            self,
            frame_observer=frame_observer,
            force_transcode=force_transcode,
        )


@dataclass(frozen=True, slots=True)
class VideoBytes:
    data: bytes = field(repr=False)
    uri: str | None = None
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None

    def open(self) -> IO[bytes]:
        return io.BytesIO(self.data)

    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoBytes":
        next_from, next_to = _compose_clip_bounds(
            current_from=self.from_timestamp_s,
            current_to=self.to_timestamp_s,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )
        return VideoBytes(
            data=self.data,
            uri=self.uri,
            from_timestamp_s=next_from,
            to_timestamp_s=next_to,
        )

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        from refiner.video.decode import iter_encoded_frames

        return iter_encoded_frames(self)

    async def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]:
        async for frame in self.iter_frames():
            yield frame.frame.to_ndarray(format="rgb24")

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]:
        from refiner.video.decode import iter_frame_windows

        return iter_frame_windows(
            self,
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )

    async def write_to(
        self,
        writer: VideoStreamWriter,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo:
        return await writer.write_encoded_video(
            self,
            frame_observer=frame_observer,
            force_transcode=force_transcode,
        )


@dataclass(frozen=True, slots=True, eq=False)
class VideoFrameSequence:
    """Video source backed by an iterable of RGB frame arrays."""

    frames: Callable[[], Iterable[Any]] | Iterable[Any] = field(
        repr=False, compare=False
    )
    fps: float = 30.0
    frame_count: int | None = None
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be > 0")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("frame_count must be >= 0")
        source = self.frames
        if not callable(source) and iter(source) is source:
            raise ValueError(
                "frames must be repeatable; pass a callable that returns a fresh iterator"
            )
        object.__setattr__(self, "fps", fps)

    @property
    def duration_s(self) -> float | None:
        if self.frame_count is None:
            return None
        return self.frame_count / float(self.fps)

    def iter_frame_arrays(self) -> Iterator[np.ndarray]:
        start_idx = (
            0
            if self.from_timestamp_s is None
            else max(0, int(math.floor(float(self.from_timestamp_s) * self.fps)))
        )
        if self.to_timestamp_s is not None:
            end_idx = max(
                start_idx, int(math.ceil(float(self.to_timestamp_s) * self.fps))
            )
        elif self.frame_count is not None:
            end_idx = start_idx + self.frame_count
        else:
            end_idx = None
        source = self.frames
        frames = (
            cast(Callable[[], Iterable[Any]], source)() if callable(source) else source
        )
        for index, frame in enumerate(frames):
            if index < start_idx:
                continue
            if end_idx is not None and index >= end_idx:
                break
            array = np.asarray(frame)
            if array.ndim != 3 or int(array.shape[2]) != 3:
                raise ValueError(
                    "video frames must have shape [height, width, channels=3]"
                )
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            yield array

    async def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]:
        for frame in self.iter_frame_arrays():
            yield frame

    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoFrameSequence":
        next_from, next_to = _compose_clip_bounds(
            current_from=self.from_timestamp_s,
            current_to=self.to_timestamp_s,
            from_timestamp_s=from_timestamp_s,
            to_timestamp_s=to_timestamp_s,
        )
        frame_count = self.frame_count
        if frame_count is not None:
            next_from_s = float(next_from or 0.0)
            view_from = float(self.from_timestamp_s or 0.0)
            view_to = (
                float(self.to_timestamp_s)
                if self.to_timestamp_s is not None
                else view_from + frame_count / float(self.fps)
            )
            if next_from_s > view_to or (
                next_to is not None and next_to > view_to + 1e-6
            ):
                raise ValueError("clip bounds are beyond the current video view")
            effective_to = view_to if next_to is None else next_to
            start_idx = max(0, int(math.floor(next_from_s * self.fps)))
            end_idx = max(start_idx, int(math.ceil(float(effective_to) * self.fps)))
            frame_count = end_idx - start_idx
        return VideoFrameSequence(
            self.frames,
            fps=self.fps,
            frame_count=frame_count,
            from_timestamp_s=next_from,
            to_timestamp_s=next_to,
        )

    async def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        import av
        from refiner.video.decode import DecodedVideoFrame

        start_idx = (
            0
            if self.from_timestamp_s is None
            else max(0, int(math.floor(float(self.from_timestamp_s) * self.fps)))
        )
        rate = Fraction(self.fps).limit_denominator(100000)
        for offset, frame_array in enumerate(self.iter_frame_arrays()):
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            frame.pts = start_idx + offset
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            yield DecodedVideoFrame(
                index=offset,
                pts=offset,
                timestamp_s=offset / float(self.fps),
                width=int(frame.width),
                height=int(frame.height),
                frame=frame,
            )

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]:
        from refiner.video.decode import iter_frame_windows

        return iter_frame_windows(
            self,
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )

    async def write_to(
        self,
        writer: VideoStreamWriter,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo:
        return await writer.write_frame_array_video(
            self,
            frame_observer=frame_observer,
        )


@dataclass(frozen=True, slots=True, eq=False)
class VideoFrameArray:
    frames: Any = field(repr=False, compare=False)
    fps: float = 30.0
    _array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be > 0")
        array = np.asarray(self.frames)
        if array.ndim != 4:
            raise ValueError(
                "video frame arrays must have shape [frames, height, width, channels]"
            )
        if int(array.shape[3]) != 3:
            raise ValueError("video frame arrays must have 3 RGB channels")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "_array", array)

    @property
    def frame_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def duration_s(self) -> float:
        return self.frame_count / float(self.fps)

    def iter_frame_arrays(self) -> Iterator[np.ndarray]:
        yield from self._array

    async def iter_numpy_frames(self) -> AsyncIterator[np.ndarray]:
        for frame in self._array:
            yield frame

    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoFrameArray":
        start_s = 0.0 if from_timestamp_s is None else float(from_timestamp_s)
        end_s = self.duration_s if to_timestamp_s is None else float(to_timestamp_s)
        if start_s < 0:
            raise ValueError("from_timestamp_s must be >= 0")
        if end_s < start_s:
            raise ValueError("to_timestamp_s must be >= from_timestamp_s")
        if start_s > self.duration_s or end_s > self.duration_s + 1e-6:
            raise ValueError("clip bounds are beyond the current video view")
        start_idx = max(0, min(self.frame_count, int(math.floor(start_s * self.fps))))
        end_idx = max(
            start_idx, min(self.frame_count, int(math.ceil(end_s * self.fps)))
        )
        return VideoFrameArray(self._array[start_idx:end_idx], fps=self.fps)

    async def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        import av
        from refiner.video.decode import DecodedVideoFrame

        for index, frame_array in enumerate(self._array):
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            frame.pts = index
            rate = Fraction(self.fps).limit_denominator(100000)
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            yield DecodedVideoFrame(
                index=index,
                pts=index,
                timestamp_s=index / float(self.fps),
                width=int(frame.width),
                height=int(frame.height),
                frame=frame,
            )

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]:
        from refiner.video.decode import iter_frame_windows

        return iter_frame_windows(
            self,
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )

    async def write_to(
        self,
        writer: VideoStreamWriter,
        *,
        frame_observer: FrameObserver | None = None,
        force_transcode: bool = False,
    ) -> WrittenVideo:
        return await writer.write_frame_array_video(
            self,
            frame_observer=frame_observer,
        )


def video_from_storage_value(
    storage: str | None,
    value: Any,
    *,
    fps: float = 30.0,
) -> VideoSource | None:
    if value is None:
        return None
    if isinstance(value, VideoSource):
        return value
    if storage in {None, "path"} and isinstance(value, str):
        return VideoFile(DataFile.resolve(value))
    if storage in {None, "bytes"} and isinstance(value, bytes):
        return VideoBytes(data=value)
    if storage in {None, "bytes_with_path"} and isinstance(value, Mapping):
        data = value.get("bytes")
        path = value.get("path")
        if isinstance(data, bytes):
            return VideoBytes(data=data, uri=path if isinstance(path, str) else None)
    if storage in {None, "frame_array"}:
        try:
            return VideoFrameArray(value, fps=fps)
        except ValueError:
            return None
    return None


__all__ = [
    "VideoBytes",
    "VideoFile",
    "VideoFrameArray",
    "VideoFrameSequence",
    "VideoSource",
    "video_from_storage_value",
]
