from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import threading
from typing import cast

import pytest

from refiner.pipeline.sinks._s3_multipart import (
    ConcurrentS3MultipartWriter,
    part_bytes_for_size,
)


_PART_BYTES = 5 << 20


class _FakeS3FileSystem:
    s3_additional_kwargs: dict[str, object] = {}

    def __init__(
        self,
        *,
        block_uploads: bool = False,
        fail_part: int | None = None,
    ) -> None:
        self.block_uploads = block_uploads
        self.fail_part = fail_part
        self.uploads: dict[int, bytes] = {}
        self.calls: defaultdict[str, int] = defaultdict(int)
        self.completed_parts: list[dict[str, object]] = []
        self.piped: bytes | None = None
        self.removed: list[str] = []
        self.invalidated: list[str] = []
        self.release_uploads = threading.Event()
        self.two_uploads_started = threading.Event()
        self._active_uploads = 0
        self._lock = threading.Lock()

    def split_path(self, path: str) -> tuple[str, str, None]:
        return "bucket", path, None

    def call_s3(self, method: str, *_args: object, **kwargs: object):
        self.calls[method] += 1
        if method == "create_multipart_upload":
            return {"UploadId": "upload-1"}
        if method == "upload_part":
            raw_part_number = kwargs["PartNumber"]
            raw_payload = kwargs["Body"]
            assert isinstance(raw_part_number, int)
            assert isinstance(raw_payload, bytes)
            part_number = raw_part_number
            payload = raw_payload
            with self._lock:
                self._active_uploads += 1
                if self._active_uploads == 2:
                    self.two_uploads_started.set()
            try:
                if self.block_uploads and not self.release_uploads.wait(timeout=2):
                    raise TimeoutError("test did not release multipart uploads")
                if part_number == self.fail_part:
                    raise RuntimeError(f"part {part_number} failed")
                self.uploads[part_number] = payload
                return {"ETag": f"etag-{part_number}"}
            finally:
                with self._lock:
                    self._active_uploads -= 1
        if method == "complete_multipart_upload":
            multipart = kwargs["MultipartUpload"]
            assert isinstance(multipart, Mapping)
            parts = cast(Mapping[str, object], multipart)["Parts"]
            assert isinstance(parts, list)
            assert all(isinstance(part, dict) for part in parts)
            self.completed_parts = cast(list[dict[str, object]], parts)
            return {}
        if method == "abort_multipart_upload":
            return {}
        raise AssertionError(f"unexpected S3 method: {method}")

    def pipe_file(self, _path: str, data: bytes, **_kwargs: object) -> None:
        self.calls["pipe_file"] += 1
        self.piped = data

    def invalidate_cache(self, path: str) -> None:
        self.invalidated.append(path)

    def rm(self, path: str) -> None:
        self.removed.append(path)


def test_concurrent_s3_writer_uses_one_put_for_a_small_blob() -> None:
    filesystem = _FakeS3FileSystem()
    writer = ConcurrentS3MultipartWriter(
        filesystem,
        "small.blob",
        part_bytes=_PART_BYTES,
        max_in_flight=2,
    )

    assert writer.write(b"small") == 5
    writer.close()

    assert filesystem.piped == b"small"
    assert filesystem.calls["create_multipart_upload"] == 0
    assert writer.closed


def test_concurrent_s3_writer_uses_one_put_at_part_boundary() -> None:
    filesystem = _FakeS3FileSystem()
    writer = ConcurrentS3MultipartWriter(
        filesystem,
        "one-part.blob",
        part_bytes=_PART_BYTES,
        max_in_flight=2,
    )
    payload = b"x" * _PART_BYTES

    writer.write(payload)
    writer.close()

    assert filesystem.piped == payload
    assert filesystem.calls["create_multipart_upload"] == 0


def test_concurrent_s3_writer_uploads_fixed_parts_in_order() -> None:
    filesystem = _FakeS3FileSystem()
    writer = ConcurrentS3MultipartWriter(
        filesystem,
        "packed.blob",
        part_bytes=_PART_BYTES,
        max_in_flight=2,
    )
    expected = b"a" * (3 << 20) + b"b" * (4 << 20) + b"c" * (4 << 20)

    writer.write(b"a" * (3 << 20))
    writer.write(b"b" * (4 << 20))
    writer.write(b"c" * (4 << 20))
    writer.close()

    assert [
        len(filesystem.uploads[number]) for number in sorted(filesystem.uploads)
    ] == [
        5 << 20,
        5 << 20,
        1 << 20,
    ]
    assert (
        b"".join(filesystem.uploads[number] for number in sorted(filesystem.uploads))
        == expected
    )
    assert [part["PartNumber"] for part in filesystem.completed_parts] == [1, 2, 3]
    assert filesystem.invalidated == ["packed.blob"]


def test_concurrent_s3_writer_runs_parts_concurrently() -> None:
    filesystem = _FakeS3FileSystem(block_uploads=True)
    writer = ConcurrentS3MultipartWriter(
        filesystem,
        "concurrent.blob",
        part_bytes=_PART_BYTES,
        max_in_flight=3,
    )
    close_error: list[BaseException] = []

    writer.write(b"x" * (3 * _PART_BYTES))

    def close_writer() -> None:
        try:
            writer.close()
        except BaseException as error:
            close_error.append(error)

    thread = threading.Thread(target=close_writer)
    thread.start()
    assert filesystem.two_uploads_started.wait(timeout=2)
    filesystem.release_uploads.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert close_error == []
    assert filesystem.calls["upload_part"] == 3
    assert filesystem.calls["complete_multipart_upload"] == 1


def test_concurrent_s3_writer_aborts_after_part_failure() -> None:
    filesystem = _FakeS3FileSystem(fail_part=2)
    writer = ConcurrentS3MultipartWriter(
        filesystem,
        "failed.blob",
        part_bytes=_PART_BYTES,
        max_in_flight=2,
    )

    try:
        writer.write(b"x" * (2 * _PART_BYTES))
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError, match="part 2 failed"):
        writer.close()

    assert filesystem.calls["abort_multipart_upload"] == 1
    assert filesystem.calls["complete_multipart_upload"] == 0
    assert filesystem.removed == ["failed.blob"]
    assert writer.closed


def test_concurrent_s3_writer_validates_bounds() -> None:
    filesystem = _FakeS3FileSystem()
    with pytest.raises(ValueError, match="at least 5 MiB"):
        ConcurrentS3MultipartWriter(filesystem, "x", part_bytes=(5 << 20) - 1)
    with pytest.raises(ValueError, match="max_in_flight must be > 0"):
        ConcurrentS3MultipartWriter(filesystem, "x", max_in_flight=0)


def test_s3_part_size_stays_within_object_part_limit() -> None:
    assert part_bytes_for_size(1 << 30) == 64 << 20
    maximum_part_bytes = part_bytes_for_size(5 << 40)
    assert maximum_part_bytes == ((5 << 40) + 9_999) // 10_000
    assert maximum_part_bytes * 10_000 >= 5 << 40
    with pytest.raises(ValueError, match="5 TiB"):
        part_bytes_for_size((5 << 40) + 1)
