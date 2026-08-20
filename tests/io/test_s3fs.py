from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import threading
from typing import cast

import pytest
import s3fs

from refiner.io._s3fs import (
    install_s3fs_concurrent_writes,
    is_s3fs,
    part_bytes_for_size,
)

install_s3fs_concurrent_writes()
ConcurrentS3File = s3fs.core.S3File


_PART_BYTES = 5 << 20


class _S3Recorder:
    def __init__(
        self,
        *,
        block_uploads: bool = False,
        fail_part: int | None = None,
        fail_completion: bool = False,
    ) -> None:
        self.block_uploads = block_uploads
        self.fail_part = fail_part
        self.fail_completion = fail_completion
        self.uploads: dict[int, bytes] = {}
        self.calls: defaultdict[str, int] = defaultdict(int)
        self.completed_parts: list[dict[str, object]] = []
        self.put_payload: bytes | None = None
        self.release_uploads = threading.Event()
        self.two_uploads_started = threading.Event()
        self._active_uploads = 0
        self._lock = threading.Lock()

    def __call__(self, method: str, *_args: object, **kwargs: object) -> dict:
        self.calls[method] += 1
        if method == "create_multipart_upload":
            return {"UploadId": "upload-1"}
        if method == "upload_part":
            part_number = cast(int, kwargs["PartNumber"])
            payload = cast(bytes, kwargs["Body"])
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
            if self.fail_completion:
                raise RuntimeError("completion failed")
            multipart = cast(Mapping[str, object], kwargs["MultipartUpload"])
            self.completed_parts = cast(
                list[dict[str, object]],
                multipart["Parts"],
            )
            return {}
        if method == "abort_multipart_upload":
            return {}
        if method == "put_object":
            self.put_payload = cast(bytes, kwargs["Body"])
            return {}
        raise AssertionError(f"unexpected S3 method: {method}")


def _filesystem(recorder: _S3Recorder) -> s3fs.S3FileSystem:
    filesystem = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    setattr(filesystem, "_refiner_multipart_part_bytes", _PART_BYTES)
    setattr(filesystem, "_refiner_multipart_max_in_flight", 2)
    filesystem.call_s3 = recorder
    filesystem.abort_mpu = lambda bucket, key, upload_id: recorder(
        "abort_multipart_upload",
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
    )
    return filesystem


def test_patch_replaces_whole_s3_file_class_idempotently() -> None:
    assert install_s3fs_concurrent_writes()
    first = s3fs.core.S3File
    assert install_s3fs_concurrent_writes()

    assert first is ConcurrentS3File
    assert s3fs.core.S3File is first
    assert s3fs.S3File is first


def test_s3fs_subclasses_are_detected() -> None:
    class CustomS3FileSystem(s3fs.S3FileSystem):
        pass

    assert is_s3fs(CustomS3FileSystem(anon=True, skip_instance_cache=True))


def test_regular_s3fs_open_constructs_the_concurrent_file() -> None:
    recorder = _S3Recorder()
    filesystem = _filesystem(recorder)

    writer = filesystem.open("bucket/packed.blob", "wb", size=3 * _PART_BYTES)
    try:
        assert type(writer) is ConcurrentS3File
        assert writer.blocksize == _PART_BYTES
        writer.write(b"x")
    finally:
        writer.close()


def test_explicit_s3fs_block_size_remains_the_part_size() -> None:
    recorder = _S3Recorder()
    filesystem = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    filesystem.call_s3 = recorder

    with filesystem.open(
        "bucket/custom-block.blob",
        "wb",
        block_size=6 << 20,
        size=1,
    ) as writer:
        assert writer.blocksize == 6 << 20
        writer.write(b"x")


def test_s3fs_filesystem_default_remains_the_part_size() -> None:
    recorder = _S3Recorder()
    filesystem = s3fs.S3FileSystem(
        anon=True,
        default_block_size=_PART_BYTES,
        skip_instance_cache=True,
    )
    filesystem.call_s3 = recorder

    with filesystem.open(
        "bucket/custom-default.blob",
        "wb",
        size=1,
    ) as writer:
        assert writer.blocksize == _PART_BYTES
        assert writer._refiner_batch_bytes == 4 * _PART_BYTES
        writer.write(b"x")


def test_small_write_retains_s3fs_one_shot_put() -> None:
    recorder = _S3Recorder()
    filesystem = _filesystem(recorder)

    with filesystem.open("bucket/small.blob", "wb", size=5) as writer:
        assert writer.write(b"small") == 5

    assert recorder.put_payload == b"small"
    assert recorder.calls["create_multipart_upload"] == 0


def test_uploads_fixed_parts_and_completes_in_order() -> None:
    recorder = _S3Recorder()
    filesystem = _filesystem(recorder)
    expected = b"a" * (3 << 20) + b"b" * (4 << 20) + b"c" * (4 << 20)

    with filesystem.open(
        "bucket/packed.blob",
        "wb",
        size=len(expected),
    ) as writer:
        writer.write(b"a" * (3 << 20))
        writer.write(b"b" * (4 << 20))
        writer.write(b"c" * (4 << 20))

    assert [len(recorder.uploads[number]) for number in sorted(recorder.uploads)] == [
        5 << 20,
        5 << 20,
        1 << 20,
    ]
    assert (
        b"".join(recorder.uploads[number] for number in sorted(recorder.uploads))
        == expected
    )
    assert [part["PartNumber"] for part in recorder.completed_parts] == [1, 2, 3]


def test_upload_batch_runs_concurrently() -> None:
    recorder = _S3Recorder(block_uploads=True)
    filesystem = _filesystem(recorder)
    writer = filesystem.open("bucket/concurrent.blob", "wb", size=2 * _PART_BYTES)
    write_error: list[BaseException] = []

    def write_batch() -> None:
        try:
            writer.write(b"x" * (2 * _PART_BYTES))
        except BaseException as error:
            write_error.append(error)

    thread = threading.Thread(target=write_batch)
    thread.start()
    assert recorder.two_uploads_started.wait(timeout=2)
    recorder.release_uploads.set()
    thread.join(timeout=2)
    writer.close()

    assert not thread.is_alive()
    assert write_error == []
    assert recorder.calls["upload_part"] == 2
    assert recorder.calls["complete_multipart_upload"] == 1


def test_part_failure_aborts_and_closes_the_file() -> None:
    recorder = _S3Recorder(fail_part=2)
    filesystem = _filesystem(recorder)
    writer = filesystem.open("bucket/failed.blob", "wb", size=2 * _PART_BYTES)

    with pytest.raises(RuntimeError, match="part 2 failed"):
        writer.write(b"x" * (2 * _PART_BYTES))

    assert recorder.calls["abort_multipart_upload"] == 1
    assert recorder.calls["complete_multipart_upload"] == 0
    assert writer.closed


def test_completion_failure_aborts_and_closes_the_file() -> None:
    recorder = _S3Recorder(fail_completion=True)
    filesystem = _filesystem(recorder)
    writer = filesystem.open("bucket/failed.blob", "wb", size=2 * _PART_BYTES)
    writer.write(b"x" * (2 * _PART_BYTES))

    with pytest.raises(RuntimeError, match="completion failed"):
        writer.close()

    assert recorder.calls["abort_multipart_upload"] == 1
    assert recorder.calls["complete_multipart_upload"] == 1
    assert writer.closed


def test_size_hint_selects_a_part_size_within_s3_limits() -> None:
    assert part_bytes_for_size(1 << 30) == 64 << 20
    maximum_part_bytes = part_bytes_for_size(5 << 40)
    assert maximum_part_bytes == ((5 << 40) + 9_999) // 10_000
    assert maximum_part_bytes * 10_000 >= 5 << 40
    with pytest.raises(ValueError, match="5 TiB"):
        part_bytes_for_size((5 << 40) + 1)


def test_read_files_keep_normal_s3fs_buffering() -> None:
    recorder = _S3Recorder()
    filesystem = _filesystem(recorder)

    reader = filesystem.open("bucket/input", "rb", block_size=1234, size=0)
    try:
        assert type(reader) is ConcurrentS3File
        assert reader.blocksize == 1234
        assert not hasattr(reader, "_refiner_part_bytes")
    finally:
        reader.close()
