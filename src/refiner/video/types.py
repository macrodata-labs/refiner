from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from refiner.io import DataFile
from refiner.video.arrays import coerce_frame_array

if TYPE_CHECKING:
    from refiner.video.decode import DecodedFrameWindow, DecodedVideoFrame
    from refiner.video.transcode import VideoTranscodeConfig


@runtime_checkable
class VideoSource(Protocol):
    def open(self) -> IO[bytes]: ...

    async def export_clip(
        self,
        *,
        force_transcode: bool = False,
        transcode_config: VideoTranscodeConfig | None = None,
    ) -> bytes: ...

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]: ...

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]: ...


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

    async def export_clip(
        self,
        *,
        force_transcode: bool = False,
        transcode_config: VideoTranscodeConfig | None = None,
    ) -> bytes:
        from refiner.video.decode import export_clip

        return await export_clip(
            self,
            force_transcode=force_transcode,
            transcode_config=transcode_config,
        )

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        from refiner.video.decode import iter_frames

        return iter_frames(self)

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


@dataclass(frozen=True, slots=True, eq=False)
class VideoFrameArray:
    frames: Any = field(repr=False, compare=False)
    fps: int = 30
    _array: np.ndarray = field(init=False, repr=False)
    _encoded_bytes: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        array = coerce_frame_array(self.frames)
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        object.__setattr__(self, "_array", array)

    @property
    def frame_count(self) -> int:
        return int(self._array.shape[0])

    @property
    def height(self) -> int:
        return int(self._array.shape[1])

    @property
    def width(self) -> int:
        return int(self._array.shape[2])

    @property
    def channels(self) -> int:
        return int(self._array.shape[3])

    @property
    def shape(self) -> tuple[int, int, int, int]:
        t, h, w, c = self._array.shape
        return int(t), int(h), int(w), int(c)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frame_arrays()

    def iter_frame_arrays(self) -> Iterator[np.ndarray]:
        for index in range(self.frame_count):
            yield self._array[index]

    def open(self) -> IO[bytes]:
        from refiner.video.encode import open_video_bytes

        return open_video_bytes(self.to_bytes())

    def to_bytes(self) -> bytes:
        from refiner.video.encode import encode_frame_array_to_mp4

        encoded = self._encoded_bytes
        if encoded is None:
            encoded = encode_frame_array_to_mp4(self._array, fps=self.fps)
            object.__setattr__(self, "_encoded_bytes", encoded)
        return encoded

    async def export_clip(
        self,
        *,
        force_transcode: bool = False,
        transcode_config: VideoTranscodeConfig | None = None,
    ) -> bytes:
        from refiner.video.decode import export_clip

        return await export_clip(
            self,
            force_transcode=force_transcode,
            transcode_config=transcode_config,
        )

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        from refiner.video.decode import iter_frames

        return iter_frames(self)

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


__all__ = ["VideoFrameArray", "VideoFile", "VideoSource"]
