from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
import queue
import threading
from typing import Any, BinaryIO, cast

from refiner.io.datafile import DataFile

DEFAULT_BLOB_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_BLOB_STREAM_PREFETCH_CHUNKS = 4
_QUEUE_POLL_SECONDS = 0.05
_CLOSE_JOIN_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class _BlobRange:
    path: str
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class _ProducerFailure:
    error: BaseException


class _EndOfBlob:
    pass


_END_OF_BLOB = _EndOfBlob()
_QueueItem = bytes | _ProducerFailure | _EndOfBlob


def _parse_blob_reference(reference: Mapping[str, object]) -> _BlobRange:
    path = reference.get("path")
    offset = reference.get("offset")
    size = reference.get("size")
    if not isinstance(path, str) or not path:
        raise ValueError("blob path must be a non-empty string")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("blob offset must be a non-negative integer")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("blob size must be a non-negative integer")
    return _BlobRange(path=path, offset=offset, size=size)


def _short_blob_error(blob: _BlobRange, received: int) -> EOFError:
    return EOFError(
        f"Blob reference requested {blob.size} bytes at offset {blob.offset}, "
        f"but only {received} bytes were available"
    )


class _BlobStream(io.RawIOBase):
    """Bounded producer-consumer stream for one exact blob byte range."""

    def __init__(
        self,
        blob: _BlobRange,
        *,
        chunk_bytes: int,
        prefetch_chunks: int,
    ) -> None:
        super().__init__()
        self._blob = blob
        self._chunk_bytes = chunk_bytes
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=prefetch_chunks)
        self._cancelled = threading.Event()
        self._current = memoryview(b"")
        self._eof = False
        self._thread = threading.Thread(
            target=self._produce,
            name=f"refiner-blob-stream-{id(self):x}",
            daemon=True,
        )
        self._thread.start()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer: Any) -> int:
        self._checkClosed()
        output = memoryview(buffer).cast("B")
        if not output or self._eof:
            return 0

        written = 0
        while written < len(output):
            if self._current:
                count = min(len(output) - written, len(self._current))
                output[written : written + count] = self._current[:count]
                self._current = self._current[count:]
                written += count
                continue

            item = self._queue.get()
            if isinstance(item, bytes):
                self._current = memoryview(item)
                continue
            if isinstance(item, _ProducerFailure):
                if written:
                    self._queue.put_nowait(item)
                    return written
                raise item.error

            self._eof = True
            break
        return written

    def close(self) -> None:
        if self.closed:
            return
        self._cancelled.set()
        # A storage backend can block indefinitely inside open() or read(). The
        # daemon producer will close its source when that call returns, but a
        # consumer closing early must not be held hostage by the backend.
        self._thread.join(timeout=_CLOSE_JOIN_SECONDS)
        self._current = memoryview(b"")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        super().close()

    def _produce(self) -> None:
        try:
            with DataFile.resolve(self._blob.path).open(
                "rb", cache_type="none"
            ) as source:
                source.seek(self._blob.offset)
                received = 0
                while received < self._blob.size and not self._cancelled.is_set():
                    requested = min(self._chunk_bytes, self._blob.size - received)
                    data = bytes(source.read(requested))
                    if len(data) > requested:
                        raise OSError(
                            "Storage backend returned bytes outside the requested blob range"
                        )
                    if not data:
                        raise _short_blob_error(self._blob, received)
                    if not self._put(data):
                        return
                    received += len(data)
            if not self._cancelled.is_set():
                self._put(_END_OF_BLOB)
        except BaseException as error:
            if not self._cancelled.is_set():
                self._put(_ProducerFailure(error))

    def _put(self, item: _QueueItem) -> bool:
        while not self._cancelled.is_set():
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False


def open_blob_stream(
    reference: Mapping[str, object],
    *,
    chunk_bytes: int = DEFAULT_BLOB_STREAM_CHUNK_BYTES,
    prefetch_chunks: int = DEFAULT_BLOB_STREAM_PREFETCH_CHUNKS,
) -> BinaryIO:
    """Open a bounded, non-seekable stream over a Refiner blob reference.

    The returned stream owns one producer thread and must be closed, preferably
    by using it as a context manager. Reads never expose bytes outside the
    reference's exact ``offset`` and ``size`` range.
    """
    blob = _parse_blob_reference(reference)
    if (
        not isinstance(chunk_bytes, int)
        or isinstance(chunk_bytes, bool)
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer")
    if (
        not isinstance(prefetch_chunks, int)
        or isinstance(prefetch_chunks, bool)
        or prefetch_chunks <= 0
    ):
        raise ValueError("prefetch_chunks must be a positive integer")
    return cast(
        BinaryIO,
        _BlobStream(
            blob,
            chunk_bytes=chunk_bytes,
            prefetch_chunks=prefetch_chunks,
        ),
    )


def read_blob(reference: Mapping[str, object]) -> bytes:
    """Read the exact byte range described by a Refiner blob reference."""
    blob = _parse_blob_reference(reference)

    chunks: list[bytes] = []
    remaining = blob.size
    with DataFile.resolve(blob.path).open("rb") as stream:
        stream.seek(blob.offset)
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    if remaining:
        raise _short_blob_error(blob, blob.size - remaining)
    return b"".join(chunks)


__all__ = [
    "DEFAULT_BLOB_STREAM_CHUNK_BYTES",
    "DEFAULT_BLOB_STREAM_PREFETCH_CHUNKS",
    "open_blob_stream",
    "read_blob",
]
