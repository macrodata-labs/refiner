from __future__ import annotations

from typing import IO

from refiner.io import DataFile
from refiner.pipeline.utils.cache.file_cache import _CacheFileLease, get_media_cache


class MediaFile:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._lease: _CacheFileLease | None = None
        self._bytes_cache: bytes | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        if self._lease is not None:
            return open(self._lease.path, mode=mode)
        return self._data_file.open(mode=mode)

    async def cache_bytes(
        self,
        *,
        suffix: str | None = None,
        cache_name: str = "default",
    ) -> bytes:
        del suffix
        if self._bytes_cache is not None:
            return self._bytes_cache

        if self._data_file.is_local:
            with self.open("rb") as f:
                self._bytes_cache = f.read()
            return self._bytes_cache

        cache = get_media_cache(cache_name)
        async with cache.cached(file=self._data_file) as local_path:
            self._bytes_cache = self._read_bytes(local_path=local_path)
        return self._bytes_cache

    def _read_bytes(self, *, local_path: str) -> bytes:
        with open(local_path, "rb") as f:
            return f.read()

    async def cache_file(self, *, cache_name: str = "default") -> str:
        if self._lease is not None:
            return self._lease.path

        cache = get_media_cache(cache_name)
        self._lease = await cache.acquire_file_lease(self._data_file)
        return self._lease.path

    def cleanup(self) -> None:
        if self._lease is not None:
            self._lease.release()
        self._lease = None
        self._bytes_cache = None

    @property
    def bytes_cache(self) -> bytes | None:
        return self._bytes_cache


__all__ = ["MediaFile"]
