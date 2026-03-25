from __future__ import annotations

from dataclasses import dataclass
from typing import IO

from refiner.io import DataFile


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


__all__ = ["VideoFile"]
