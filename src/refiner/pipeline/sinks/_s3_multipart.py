from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any


_MIN_PART_BYTES = 5 << 20
_MAX_PART_BYTES = 5 << 30
_MAX_PARTS = 10_000
DEFAULT_PART_BYTES = 64 << 20
DEFAULT_MAX_IN_FLIGHT = 4


def is_s3fs(filesystem: object) -> bool:
    """Return whether *filesystem* is an s3fs filesystem without importing s3fs."""
    return type(filesystem).__module__.split(".", maxsplit=1)[0] == "s3fs" and all(
        hasattr(filesystem, name) for name in ("call_s3", "pipe_file", "split_path")
    )


def part_bytes_for_size(size_hint: int) -> int:
    """Choose a fixed part size that stays within S3's 10,000-part limit."""
    if size_hint > 5 << 40:
        raise ValueError("S3 blob size exceeds the 5 TiB multipart object limit")
    part_bytes = max(DEFAULT_PART_BYTES, (size_hint + _MAX_PARTS - 1) // _MAX_PARTS)
    if part_bytes > _MAX_PART_BYTES:
        raise ValueError("S3 blob size exceeds the 5 TiB multipart object limit")
    return part_bytes


class ConcurrentS3MultipartWriter:
    """A bounded, fixed-part multipart writer for S3-compatible object stores."""

    def __init__(
        self,
        filesystem: Any,
        path: str,
        *,
        part_bytes: int = DEFAULT_PART_BYTES,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        if part_bytes < _MIN_PART_BYTES:
            raise ValueError("S3 multipart parts must be at least 5 MiB")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")
        self._fs = filesystem
        self._path = path
        self._part_bytes = part_bytes
        self._max_in_flight = max_in_flight
        self._buffer = bytearray()
        self._pending_parts: list[tuple[int, bytes]] = []
        self._parts: list[dict[str, object]] = []
        self._bucket: str | None = None
        self._key: str | None = None
        self._upload_id: str | None = None
        self._next_part_number = 1
        self._position = 0
        self._error: BaseException | None = None
        self._closed = False
        self._removed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self._position

    def writable(self) -> bool:
        return not self._closed

    def write(self, data: bytes) -> int:
        if self._closed:
            raise ValueError("write to closed file")
        if self._error is not None:
            raise self._error
        if not isinstance(data, bytes):
            raise TypeError("a bytes-like object is required")
        data_size = len(data)
        offset = 0
        if data_size and len(self._buffer) == self._part_bytes:
            self._submit_part(bytes(self._buffer))
            self._buffer.clear()
        while offset < data_size:
            remaining = data_size - offset
            if not self._buffer and remaining > self._part_bytes:
                end = offset + self._part_bytes
                self._submit_part(data[offset:end])
                offset = end
                continue
            take = min(self._part_bytes - len(self._buffer), remaining)
            self._buffer.extend(data[offset : offset + take])
            offset += take
            if len(self._buffer) == self._part_bytes and offset < data_size:
                self._submit_part(bytes(self._buffer))
                self._buffer.clear()
        self._position += data_size
        return data_size

    def _s3_kwargs(self) -> dict[str, object]:
        return dict(getattr(self._fs, "s3_additional_kwargs", None) or {})

    def _ensure_multipart_upload(self) -> None:
        if self._upload_id is not None:
            return
        bucket, key, _version_id = self._fs.split_path(self._path)
        response = self._fs.call_s3(
            "create_multipart_upload",
            self._s3_kwargs(),
            Bucket=bucket,
            Key=key,
        )
        self._bucket = bucket
        self._key = key
        self._upload_id = response["UploadId"]

    def _submit_part(self, payload: bytes) -> None:
        self._ensure_multipart_upload()
        part_number = self._next_part_number
        self._next_part_number += 1
        self._pending_parts.append((part_number, payload))
        if len(self._pending_parts) == self._max_in_flight:
            self._upload_pending_parts()

    def _upload_part(self, part_number: int, payload: bytes) -> dict[str, object]:
        assert self._bucket is not None
        assert self._key is not None
        assert self._upload_id is not None
        response = self._fs.call_s3(
            "upload_part",
            self._s3_kwargs(),
            Bucket=self._bucket,
            Key=self._key,
            UploadId=self._upload_id,
            PartNumber=part_number,
            Body=payload,
        )
        part: dict[str, object] = {
            "PartNumber": part_number,
            "ETag": response["ETag"],
        }
        if checksum := response.get("ChecksumSHA256"):
            part["ChecksumSHA256"] = checksum
        return part

    def _upload_pending_parts(self) -> None:
        if not self._pending_parts:
            return
        pending = self._pending_parts
        try:
            with ThreadPoolExecutor(
                max_workers=len(pending),
                thread_name_prefix="refiner-s3-part",
            ) as executor:
                uploaded = list(
                    executor.map(
                        lambda item: self._upload_part(*item),
                        pending,
                    )
                )
        except BaseException as error:
            self._pending_parts = []
            if self._error is None:
                self._error = error
            self._abort_multipart_upload()
            self._remove_output()
            raise
        self._parts.extend(uploaded)
        self._pending_parts = []

    def _abort_multipart_upload(self) -> None:
        if self._upload_id is None:
            return
        assert self._bucket is not None
        assert self._key is not None
        with suppress(Exception):
            self._fs.call_s3(
                "abort_multipart_upload",
                Bucket=self._bucket,
                Key=self._key,
                UploadId=self._upload_id,
            )
        self._upload_id = None

    def _remove_output(self) -> None:
        if self._removed:
            return
        try:
            self._fs.rm(self._path)
        except FileNotFoundError:
            self._removed = True
        except OSError:
            pass
        else:
            self._removed = True

    def close(self) -> None:
        if self._closed:
            if self._error is not None:
                raise self._error
            return
        try:
            if self._error is not None:
                raise self._error
            if self._upload_id is None:
                self._fs.pipe_file(
                    self._path,
                    bytes(self._buffer),
                    **self._s3_kwargs(),
                )
            else:
                if self._buffer:
                    self._submit_part(bytes(self._buffer))
                self._upload_pending_parts()
                assert self._bucket is not None
                assert self._key is not None
                assert self._upload_id is not None
                self._fs.call_s3(
                    "complete_multipart_upload",
                    self._s3_kwargs(),
                    Bucket=self._bucket,
                    Key=self._key,
                    UploadId=self._upload_id,
                    MultipartUpload={"Parts": self._parts},
                )
                self._upload_id = None
                self._fs.invalidate_cache(self._path)
        except BaseException as error:
            if self._error is None:
                self._error = error
            self._abort_multipart_upload()
            self._remove_output()
            raise
        finally:
            self._buffer.clear()
            self._pending_parts = []
            self._closed = True

    def discard(self) -> None:
        if self._closed:
            return
        self._abort_multipart_upload()
        self._remove_output()
        self._buffer.clear()
        self._pending_parts = []
        self._closed = True


__all__ = [
    "ConcurrentS3MultipartWriter",
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_PART_BYTES",
    "is_s3fs",
    "part_bytes_for_size",
]
