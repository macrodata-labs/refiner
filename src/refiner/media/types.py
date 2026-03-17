from __future__ import annotations

from typing import IO

from refiner.io import DataFile


class MediaFile:
    uri: str
    _data_file: DataFile

    def __init__(self, uri: str) -> None:
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "_data_file", DataFile.resolve(uri))

    def open(self, mode: str = "rb") -> IO[bytes]:
        return self._data_file.open(mode=mode)


__all__ = ["MediaFile"]
