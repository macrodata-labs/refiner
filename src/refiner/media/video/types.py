from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from refiner.io import DataFile
from refiner.media.types import MediaFile
from refiner.pipeline.utils.cache.file_cache import _CacheFileLease


@dataclass(frozen=True, slots=True)
class VideoFile(MediaFile):
    uri: str
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None
    _data_file: DataFile = field(init=False, repr=False, compare=False)
    _lease: _CacheFileLease | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data_file", DataFile.resolve(self.uri))
        object.__setattr__(self, "_lease", None)


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: tuple[np.ndarray, ...]
    fps: int
    original_file: VideoFile
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = "rgb24"


Video: TypeAlias = VideoFile | DecodedVideo


__all__ = ["VideoFile", "Video", "DecodedVideo"]
