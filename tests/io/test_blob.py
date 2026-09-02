from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import io
import threading
from typing import Any

from fsspec.implementations.memory import MemoryFileSystem
import pytest

from refiner import open_blob_stream
from refiner.io.datafile import DataFile


class RecordingMemoryFileSystem(MemoryFileSystem):
    cachable = False

    def __init__(
        self,
        *,
        block_reads: bool = False,
        failure: BaseException | None = None,
        max_read_bytes: int | None = None,
    ) -> None:
        super().__init__(skip_instance_cache=True)
        self.block_reads = block_reads
        self.failure = failure
        self.max_read_bytes = max_read_bytes
        self.read_started = threading.Event()
        self.release_reads = threading.Event()
        self.ranges: list[tuple[int | None, int | None]] = []
        self.second_read = threading.Event()
        self.source_closed = threading.Event()

    def open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        compression=None,
        **kwargs,
    ):
        source = super().open(
            path,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )
        if mode != "rb":
            return source
        return _RecordingReader(source, self)


class _RecordingReader:
    def __init__(self, source: Any, fs: RecordingMemoryFileSystem) -> None:
        self._source = source
        self._fs = fs

    def __enter__(self) -> _RecordingReader:
        self._source.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._source.__exit__(*args)
        self._fs.source_closed.set()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._source.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        if self._fs.block_reads:
            self._fs.read_started.set()
            self._fs.release_reads.wait()
        if self._fs.max_read_bytes is not None:
            size = (
                min(size, self._fs.max_read_bytes)
                if size >= 0
                else self._fs.max_read_bytes
            )
        start = self._source.tell()
        self._fs.ranges.append((start, None if size < 0 else start + size))
        if len(self._fs.ranges) >= 2:
            self._fs.second_read.set()
        if self._fs.failure is not None:
            raise self._fs.failure
        return self._source.read(size)


@contextmanager
def use_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    *,
    block_reads: bool = False,
    failure: BaseException | None = None,
    max_read_bytes: int | None = None,
) -> Iterator[RecordingMemoryFileSystem]:
    fs = RecordingMemoryFileSystem(
        block_reads=block_reads,
        failure=failure,
        max_read_bytes=max_read_bytes,
    )
    fs.pipe_file("blocks/data.bin", data)
    data_file = DataFile(fs=fs, path="blocks/data.bin")
    monkeypatch.setattr(DataFile, "resolve", lambda path: data_file)
    yield fs


def test_open_blob_stream_reads_only_exact_ranges(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"prefix-payload-suffix") as fs:
        with open_blob_stream(
            {"path": "unused", "offset": 7, "size": 7},
            chunk_bytes=3,
            prefetch_chunks=2,
        ) as stream:
            assert stream.read() == b"payload"

    assert fs.ranges == [(7, 10), (10, 13), (13, 14)]
    assert fs.source_closed.is_set()


def test_open_blob_stream_reads_local_datafile(tmp_path) -> None:
    block = tmp_path / "block.bin"
    block.write_bytes(b"prefix-payload-suffix")

    with open_blob_stream(
        {"path": str(block), "offset": 7, "size": 7},
        chunk_bytes=2,
    ) as stream:
        assert stream.read() == b"payload"


@pytest.mark.parametrize(
    ("offset", "size", "expected"),
    [(2, 1, b"c"), (5, 0, b"")],
)
def test_open_blob_stream_reads_small_and_empty_blobs(
    monkeypatch, offset: int, size: int, expected: bytes
) -> None:
    with use_filesystem(monkeypatch, b"abcdef") as fs:
        with open_blob_stream(
            {"path": "unused", "offset": offset, "size": size},
            chunk_bytes=4,
        ) as stream:
            assert stream.read() == expected
            assert stream.read() == b""

    assert fs.ranges == ([] if size == 0 else [(offset, offset + size)])


def test_open_blob_stream_supports_partial_consumer_reads(monkeypatch) -> None:
    payload = bytes(range(31))
    with use_filesystem(monkeypatch, b"xx" + payload + b"yy"):
        with open_blob_stream(
            {"path": "unused", "offset": 2, "size": len(payload)},
            chunk_bytes=5,
            prefetch_chunks=2,
        ) as stream:
            parts = [stream.read(size) for size in (1, 3, 2, 9, 4, 12)]
            parts.append(stream.read(1))

    assert b"".join(parts) == payload
    assert [len(part) for part in parts] == [1, 3, 2, 9, 4, 12, 0]


def test_open_blob_stream_supports_partial_producer_reads(monkeypatch) -> None:
    with use_filesystem(
        monkeypatch,
        b"xxpayloadyy",
        max_read_bytes=2,
    ) as fs:
        with open_blob_stream(
            {"path": "unused", "offset": 2, "size": 7},
            chunk_bytes=5,
        ) as stream:
            assert stream.read() == b"payload"

    assert fs.ranges == [(2, 4), (4, 6), (6, 8), (8, 9)]


def test_open_blob_stream_early_close_stops_blocked_producer(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"abcdefgh") as fs:
        stream = open_blob_stream(
            {"path": "unused", "offset": 0, "size": 8},
            chunk_bytes=1,
            prefetch_chunks=1,
        )
        thread = getattr(stream, "_thread")
        assert fs.second_read.wait(timeout=1)
        assert fs.ranges == [(0, 1), (1, 2)]

        stream.close()

    assert not thread.is_alive()
    assert fs.ranges == [(0, 1), (1, 2)]
    assert fs.source_closed.is_set()


def test_open_blob_stream_close_does_not_wait_for_blocked_storage(monkeypatch) -> None:
    monkeypatch.setattr("refiner.io.blob._CLOSE_JOIN_SECONDS", 0.01)
    with use_filesystem(monkeypatch, b"payload", block_reads=True) as fs:
        stream = open_blob_stream({"path": "unused", "offset": 0, "size": 7})
        producer = getattr(stream, "_thread")
        assert fs.read_started.wait(timeout=1)

        closer = threading.Thread(target=stream.close)
        closer.start()
        closer.join(timeout=1)
        try:
            assert not closer.is_alive()
            assert producer.is_alive()
        finally:
            fs.release_reads.set()
            closer.join(timeout=1)
            producer.join(timeout=1)

    assert not producer.is_alive()
    assert fs.source_closed.is_set()


def test_open_blob_stream_rejects_short_underlying_object(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"abc"):
        with open_blob_stream(
            {"path": "unused", "offset": 1, "size": 5},
            chunk_bytes=4,
        ) as stream:
            with pytest.raises(EOFError, match="only 2 bytes were available"):
                stream.read()


def test_open_blob_stream_propagates_producer_failure(monkeypatch) -> None:
    failure = RuntimeError("storage unavailable")
    with use_filesystem(monkeypatch, b"payload", failure=failure):
        with open_blob_stream({"path": "unused", "offset": 0, "size": 7}) as stream:
            with pytest.raises(RuntimeError, match="storage unavailable") as exc_info:
                stream.read(1)

    assert exc_info.value is failure


def test_open_blob_stream_repeated_eof_reads(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"payload"):
        with open_blob_stream(
            {"path": "unused", "offset": 0, "size": 7},
            chunk_bytes=2,
        ) as stream:
            assert stream.read() == b"payload"
            assert stream.read() == b""
            assert stream.read(100) == b""
            buffer = bytearray(3)
            assert getattr(stream, "readinto")(buffer) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_bytes": 0}, "chunk_bytes must be a positive integer"),
        ({"chunk_bytes": True}, "chunk_bytes must be a positive integer"),
        ({"prefetch_chunks": 0}, "prefetch_chunks must be a positive integer"),
        ({"prefetch_chunks": False}, "prefetch_chunks must be a positive integer"),
    ],
)
def test_open_blob_stream_validates_buffering_options(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        open_blob_stream({"path": "unused", "offset": 0, "size": 1}, **kwargs)


def test_open_blob_stream_is_non_seekable(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"payload"):
        with open_blob_stream({"path": "unused", "offset": 0, "size": 7}) as stream:
            assert not stream.seekable()
            with pytest.raises(OSError):
                stream.seek(0)


def test_open_blob_stream_does_not_leave_thread_after_context_exit(monkeypatch) -> None:
    with use_filesystem(monkeypatch, b"payload"):
        with open_blob_stream({"path": "unused", "offset": 0, "size": 7}) as stream:
            thread = getattr(stream, "_thread")
            assert stream.read(1) == b"p"

    assert not thread.is_alive()
