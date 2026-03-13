from __future__ import annotations

import os
from contextlib import contextmanager
from typing import IO, Iterator, Literal

from refiner.io import DataFile
from refiner.media.cache import get_media_cache
from refiner.media.video.utils import slice_video_to_mp4_bytes
from refiner.media.cache import _CacheFileLease


class MediaFile:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._local_path: str | None = None
        self._lease: _CacheFileLease | None = None
        self._bytes_cache: bytes | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        if self._local_path is not None and os.path.exists(self._local_path):
            return open(self._local_path, mode=mode)
        if self._lease is not None:
            return open(self._lease.path, mode=mode)
        return self._data_file.open(mode=mode)

    async def cache_bytes(
        self, *, suffix: str | None = None, cache_name: str = "default"
    ) -> bytes:
        if self._bytes_cache is not None:
            return self._bytes_cache

        if self._data_file.is_local:
            with self.open("rb") as f:
                self._bytes_cache = f.read()
            return self._bytes_cache

        cache = get_media_cache(cache_name)
        async with cache.cached(
            file=self._data_file,
        ) as local_path:
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
        self._local_path = None
        self._bytes_cache = None

    @property
    def local_path(self) -> str | None:
        return self._local_path

    @property
    def bytes_cache(self) -> bytes | None:
        return self._bytes_cache

    def is_hydrated(self) -> bool:
        return self._bytes_cache is not None


__all__ = ["MediaFile"]
