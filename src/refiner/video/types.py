from __future__ import annotations

import io
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import IO, TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from refiner.io import DataFile

if TYPE_CHECKING:
    from refiner.video.decode import DecodedFrameWindow, DecodedVideoFrame
    from refiner.video.transcode import FrameObserver
    from refiner.video.writer import VideoStreamWriter, WrittenVideo


@runtime_checkable
class VideoSource(Protocol):
    def clipped(
        self,
        *,
        from_timestamp_s: float | None = None,
        to_timestamp_s: float | None = None,
    ) -> "VideoSource": ...

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]: ...

    def iter_frame_arrays(self) -> AsyncIterator[np.ndarray]: ...

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

    async def iter_frame_arrays(self) -> AsyncIterator[np.ndarray]:
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

    async def iter_frame_arrays(self) -> AsyncIterator[np.ndarray]:
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
class VideoFrameArray:
    frames: Any = field(repr=False, compare=False)
    fps: int = 30
    _array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.fps) <= 0:
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
        object.__setattr__(self, "fps", int(self.fps))
        object.__setattr__(self, "_array", array)

    @property
    def frame_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def duration_s(self) -> float:
        return self.frame_count / float(self.fps)

    @property
    def frame_arrays(self) -> np.ndarray:
        return self._array

    async def iter_frame_arrays(self) -> AsyncIterator[np.ndarray]:
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
            frame.time_base = Fraction(1, self.fps)
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
    fps: int = 30,
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
    "VideoSource",
    "video_from_storage_value",
]
