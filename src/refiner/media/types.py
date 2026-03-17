from __future__ import annotations

from typing import IO

from refiner.io import DataFile

_CONTAINER_BLOCK_SIZE = 256 * 1024


class MediaFile:
    uri: str
    _data_file: DataFile

    def __init__(self, uri: str) -> None:
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "_data_file", DataFile.resolve(uri))

    def open(self, mode: str = "rb") -> IO[bytes]:
        return self._data_file.open(mode=mode)

    def open_for_container(self, mode: str = "rb") -> IO[bytes]:
        if "r" not in mode or self._data_file.is_local:
            return self._data_file.open(mode=mode)
        return self._data_file.open(mode=mode, block_size=_CONTAINER_BLOCK_SIZE)


__all__ = ["MediaFile"]
