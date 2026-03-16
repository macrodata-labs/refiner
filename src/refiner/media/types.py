from __future__ import annotations

from typing import IO, TYPE_CHECKING

from refiner.io import DataFile
from refiner.pipeline.utils.cache.file_cache import get_media_cache

if TYPE_CHECKING:
    from refiner.pipeline.utils.cache.file_cache import _CacheFileLease


class MediaFile:
    uri: str
    _data_file: DataFile
    _lease: "_CacheFileLease | None"

    def __init__(self, uri: str) -> None:
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "_data_file", DataFile.resolve(uri))
        object.__setattr__(self, "_lease", None)

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
        lease = await cache.acquire_file_lease(self._data_file)
        object.__setattr__(self, "_lease", lease)
        return lease.path

    def cleanup(self) -> None:
        if self._lease is not None:
            self._lease.release()
        object.__setattr__(self, "_lease", None)


__all__ = ["MediaFile"]
