from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import IO

from refiner.io import DataFile
from refiner.video.decode import (
    DecodedFrameWindow,
    DecodedVideoFrame,
    export_clip,
    iter_frame_windows,
    iter_frames,
)
from refiner.video.transcode import VideoTranscodeConfig


@dataclass(frozen=True, slots=True)
class VideoFile:
    data_file: DataFile
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None

    @property
    def uri(self) -> str:
        return str(self.data_file)

    def open(self, mode: str = "rb") -> IO[bytes]:
        return self.data_file.open(mode=mode)

    async def export_clip(
        self,
        *,
        force_transcode: bool = False,
        transcode_config: VideoTranscodeConfig | None = None,
    ) -> bytes:
        return await export_clip(
            self,
            force_transcode=force_transcode,
            transcode_config=transcode_config,
        )

    def iter_frames(self) -> AsyncIterator[DecodedVideoFrame]:
        return iter_frames(self)

    def iter_frame_windows(
        self,
        *,
        offsets: Sequence[int],
        stride: int = 1,
        drop_incomplete: bool = True,
    ) -> AsyncIterator[DecodedFrameWindow]:
        return iter_frame_windows(
            self,
            offsets=offsets,
            stride=stride,
            drop_incomplete=drop_incomplete,
        )


__all__ = ["VideoFile"]
