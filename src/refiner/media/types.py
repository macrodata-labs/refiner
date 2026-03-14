from __future__ import annotations

from typing import IO

from refiner.io import DataFile
from refiner.pipeline.utils.cache.file_cache import _CacheFileLease, get_media_cache


class MediaFile:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._lease: _CacheFileLease | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        if self._lease is not None:
            return open(self._lease.path, mode=mode)
        return self._data_file.open(mode=mode)

    async def cache_file(self, *, cache_name: str = "default") -> str:
        if self._lease is not None:
            return self._lease.path
        if self._data_file.is_local:
            return str(self._data_file)

        cache = get_media_cache(cache_name)
        self._lease = await cache.acquire_file_lease(self._data_file)
        return self._lease.path

    def cleanup(self) -> None:
        if self._lease is not None:
            self._lease.release()
        self._lease = None


__all__ = ["MediaFile"]
