from __future__ import annotations

import os
from contextlib import contextmanager
from typing import IO, Iterator, Literal

from refiner.io import DataFile
from refiner.media.cache import get_media_cache
from refiner.media.video.utils import slice_video_to_mp4_bytes


class MediaFile:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._local_path: str | None = None
        self._bytes_cache: bytes | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        if self._local_path is not None and os.path.exists(self._local_path):
            return open(self._local_path, mode=mode)
        return self._data_file.open(mode=mode)

    def cache_bytes(
        self, *, suffix: str | None = None, cache_name: str = "default"
    ) -> bytes:
        if self._bytes_cache is not None:
            return self._bytes_cache

        if self._data_file.is_local:
            with self.open("rb") as f:
                self._bytes_cache = f.read()
            return self._bytes_cache

        cache = get_media_cache(cache_name)
        with cache.cached(
            file=self._data_file,
        ) as local_path:
            with open(local_path, "rb") as f:
                self._bytes_cache = f.read()
        return self._bytes_cache

    def cache_video_segment_bytes(
        self,
        *,
        from_timestamp_s: float,
        to_timestamp_s: float | None,
        cache_name: str = "default",
        extract_backend: Literal["pyav", "ffmpeg"] = "pyav",
    ) -> bytes:
        if self._bytes_cache is not None:
            return self._bytes_cache

        if self._data_file.is_local:
            local_path = self._data_file.abs_path()
            if not self._data_file.exists():
                raise FileNotFoundError(local_path)
            self._bytes_cache = slice_video_to_mp4_bytes(
                local_path=local_path,
                from_timestamp_s=from_timestamp_s,
                to_timestamp_s=to_timestamp_s,
                extract_backend=extract_backend,
            )
            return self._bytes_cache

        cache = get_media_cache(cache_name)
        with cache.cached(
            file=self._data_file,
        ) as local_path:
            self._bytes_cache = slice_video_to_mp4_bytes(
                local_path=local_path,
                from_timestamp_s=from_timestamp_s,
                to_timestamp_s=to_timestamp_s,
                extract_backend=extract_backend,
            )
        return self._bytes_cache

    @contextmanager
    def cached_path(
        self, *, suffix: str | None = None, cache_name: str = "default"
    ) -> Iterator[str]:
        self._local_path = None

        if self._data_file.is_local:
            if not self._data_file.exists():
                raise FileNotFoundError(self._data_file.abs_path())
            self._local_path = self._data_file.abs_path()
            yield self._local_path
            return

        cache = get_media_cache(cache_name)
        with cache.cached(
            file=self._data_file,
        ) as local_path:
            self._local_path = local_path
            yield local_path

    def cleanup(self) -> None:
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
