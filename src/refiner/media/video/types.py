from __future__ import annotations

from dataclasses import dataclass, field

from refiner.io import DataFile
from refiner.media.types import MediaFile


@dataclass(frozen=True, slots=True)
class VideoFile(MediaFile):
    uri: str
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None
    _data_file: DataFile = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data_file", DataFile.resolve(self.uri))


__all__ = ["VideoFile"]
