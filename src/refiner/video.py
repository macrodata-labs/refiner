from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, replace
from threading import Lock
from typing import IO

from refiner.io import DataFile


def _cleanup_temp_path(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class VideoFile:
    """Lazy, file-like video handle.

    Behavior:
        - `open()` streams directly from the underlying URI via fsspec.
        - `to_local_path()` materializes once into a temp file and registers GC cleanup.
        - `cleanup()` eagerly removes the temp file if one was created.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._data_file = DataFile.resolve(uri)
        self._lock = Lock()
        self._local_path: str | None = None
        self._cleanup: weakref.finalize | None = None

    def open(self, mode: str = "rb") -> IO[bytes]:
        return self._data_file.open(mode=mode)

    def to_local_path(self, *, suffix: str = ".mp4") -> str:
        with self._lock:
            if self._local_path is not None:
                return self._local_path

            fd, temp_path = tempfile.mkstemp(prefix="refiner_video_", suffix=suffix)
            os.close(fd)
            try:
                with self.open("rb") as src, open(temp_path, "wb") as dst:
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


@dataclass(frozen=True, slots=True)
class Video:
    """Opaque video payload handle carried in rows.

    Notes:
        - `uri` points to the fused source video file.
        - `bytes` is optional and populated by explicit hydration.
        - Decode is not implemented yet and `decode=True` is rejected.
    """

    uri: str
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
    file: VideoFile | None = None
    bytes: bytes | None = None
    decode: bool = False

    def __post_init__(self) -> None:
        if self.decode:
            raise NotImplementedError(
                "Video decoding is not implemented yet; set decode=False."
            )

    def with_bytes(self, payload: bytes | None) -> "Video":
        return replace(self, bytes=payload)

    def with_file(self, file_handle: VideoFile | None) -> "Video":
        return replace(self, file=file_handle)

    def as_file(self) -> VideoFile:
        if self.file is not None:
            return self.file
        return VideoFile(self.uri)


__all__ = ["Video", "VideoFile"]
