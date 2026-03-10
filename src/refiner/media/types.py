from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from threading import Lock
from typing import IO

from refiner.io import DataFile


def _cleanup_temp_path(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class MediaFile:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._lock = Lock()
        self._local_path: str | None = None
        self._cleanup: weakref.finalize | None = None
        self._bytes_cache: bytes | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        if self._local_path is not None:
            return open(self._local_path, mode=mode)
        return self._data_file.open(mode=mode)

    def cache_bytes(self) -> bytes:
        if self._bytes_cache is None:
            with self.open("rb") as f:
                self._bytes_cache = f.read()
        return self._bytes_cache

    def cache_locally(self, *, suffix: str | None = None) -> str:
        with self._lock:
            if self._local_path is not None:
                return self._local_path

            if self._data_file.is_local:
                if not self._data_file.exists():
                    raise FileNotFoundError(self._data_file.abs_path())
                self._local_path = self._data_file.abs_path()
                return self._local_path

            file_suffix = suffix or os.path.splitext(self._data_file.path)[1] or ".bin"
            fd, temp_path = tempfile.mkstemp(
                prefix="refiner_media_", suffix=file_suffix
            )
            os.close(fd)
            try:
                with self._data_file.open("rb") as src, open(temp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            except Exception:
                _cleanup_temp_path(temp_path)
                raise

            self._local_path = temp_path
            self._cleanup = weakref.finalize(self, _cleanup_temp_path, temp_path)
            return temp_path

    def cleanup(self) -> None:
        with self._lock:
            if self._cleanup is not None and self._cleanup.alive:
                self._cleanup()
            self._cleanup = None
            self._local_path = None

    @property
    def local_path(self) -> str | None:
        return self._local_path

    @property
    def bytes_cache(self) -> bytes | None:
        return self._bytes_cache

    def is_hydrated(self, mode: str) -> bool:
        if mode == "file":
            return self._local_path is not None
        if mode == "bytes":
            return self._bytes_cache is not None
        raise ValueError(f"Unsupported media hydration mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class Video:
    media: MediaFile
    video_key: str
    relative_path: str | None = None
    episode_index: int | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None
    chunk_index: int | None = None
    file_index: int | None = None
    fps: int | None = None
    decode: bool = False

    @property
    def uri(self) -> str:
        return self.media.uri

    def __post_init__(self) -> None:
        if self.decode:
            raise NotImplementedError(
                "Video decoding is not implemented yet; set decode=False."
            )


__all__ = ["MediaFile", "Video"]
