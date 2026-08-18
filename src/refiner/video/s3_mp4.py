from __future__ import annotations

import errno
import io
from typing import Any

_S3_MULTIPART_MIN_PART_SIZE = 5 * 1024 * 1024


def supports_s3_multipart(fs: Any) -> bool:
    """Return whether *fs* exposes s3fs' multipart-upload primitives."""
    protocol = getattr(fs, "protocol", ())
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    return (
        "s3" in protocols
        and callable(getattr(fs, "call_s3", None))
        and callable(getattr(fs, "split_path", None))
    )


class S3MultipartSeekableWriter(io.RawIOBase):
    """Bounded-memory, forward-streaming S3 writer with a mutable first part.

    FFmpeg's normal MP4 muxer writes the ``mdat`` length into the initial bytes
    only after it has written media data. S3 multipart uploads permit parts to
    arrive out of order, so this writer withholds part 1, uploads later full
    parts immediately, and uploads the patched first part during finalization.
    """

    def __init__(self, fs: Any, path: str) -> None:
        super().__init__()
        self._fs = fs
        self._path = path
        self._bucket, self._key, _ = fs.split_path(path)
        self._prefix = bytearray()
        self._tail = bytearray()
        self._position = 0
        self._size = 0
        self._upload_id: str | None = None
        self._parts: dict[int, dict[str, Any]] = {}
        self._next_media_part = 2
        self._closed = False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def tell(self) -> int:
        self._check_open()
        return self._position

    def write(self, data: Any) -> int:
        self._check_open()
        payload = bytes(data)
        end = self._position + len(payload)
        if self._position > self._size:
            raise OSError(errno.EINVAL, "cannot create holes in an MP4 output")
        if self._position < self._size and end > self._size:
            raise OSError(errno.ESPIPE, "cannot overwrite uploaded MP4 media")
        if self._position < _S3_MULTIPART_MIN_PART_SIZE:
            prefix_end = min(end, _S3_MULTIPART_MIN_PART_SIZE)
            self._write_prefix(self._position, payload[: prefix_end - self._position])
            if prefix_end < end:
                self._append_tail(payload[prefix_end - self._position :])
        elif self._position == self._size:
            self._append_tail(payload)
        else:
            raise OSError(errno.ESPIPE, "cannot overwrite uploaded MP4 media")
        self._position = end
        self._size = max(self._size, end)
        return len(payload)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported seek whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        if position > self._size:
            raise OSError(errno.EINVAL, "cannot seek beyond the MP4 output")
        if position > _S3_MULTIPART_MIN_PART_SIZE and position != self._size:
            raise OSError(errno.ESPIPE, "cannot seek into uploaded MP4 media")
        self._position = position
        return position

    def flush(self) -> None:
        # AVIO may flush while it is closing the Python file object.
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._complete()
        except Exception:
            self.abort()
            raise
        finally:
            self._closed = True
            super().close()

    def abort(self) -> None:
        if self._upload_id is not None:
            try:
                self._call_s3(
                    "abort_multipart_upload",
                    Bucket=self._bucket,
                    Key=self._key,
                    UploadId=self._upload_id,
                )
            finally:
                self._upload_id = None
                self._parts.clear()
        self._closed = True
        super().close()

    def _write_prefix(self, position: int, data: bytes) -> None:
        was_complete = len(self._prefix) == _S3_MULTIPART_MIN_PART_SIZE
        end = position + len(data)
        if end > len(self._prefix):
            if position != len(self._prefix):
                raise OSError(errno.EINVAL, "cannot create holes in MP4 prefix")
            self._prefix.extend(data)
        else:
            self._prefix[position:end] = data
        if not was_complete and len(self._prefix) == _S3_MULTIPART_MIN_PART_SIZE:
            self._ensure_upload()
            self._upload_part(part_number=1, data=bytes(self._prefix))

    def _append_tail(self, data: bytes) -> None:
        self._tail.extend(data)
        while len(self._tail) >= _S3_MULTIPART_MIN_PART_SIZE:
            self._ensure_upload()
            self._upload_part(
                part_number=self._next_media_part,
                data=bytes(self._tail[:_S3_MULTIPART_MIN_PART_SIZE]),
            )
            self._next_media_part += 1
            del self._tail[:_S3_MULTIPART_MIN_PART_SIZE]

    def _complete(self) -> None:
        if self._upload_id is None:
            self._call_s3(
                "put_object",
                Bucket=self._bucket,
                Key=self._key,
                Body=bytes(self._prefix),
            )
            return
        # Re-uploading part 1 atomically replaces its placeholder version in
        # both S3 and R2 before the multipart upload is completed.
        self._upload_part(part_number=1, data=bytes(self._prefix))
        if self._tail:
            self._upload_part(part_number=self._next_media_part, data=bytes(self._tail))
        self._call_s3(
            "complete_multipart_upload",
            Bucket=self._bucket,
            Key=self._key,
            UploadId=self._upload_id,
            MultipartUpload={
                "Parts": [self._parts[number] for number in sorted(self._parts)]
            },
        )
        self._upload_id = None
        self._parts.clear()

    def _ensure_upload(self) -> None:
        if self._upload_id is not None:
            return
        response = self._call_s3(
            "create_multipart_upload", Bucket=self._bucket, Key=self._key
        )
        self._upload_id = str(response["UploadId"])

    def _upload_part(self, *, part_number: int, data: bytes) -> None:
        if self._upload_id is None:
            raise RuntimeError("multipart upload was not initialized")
        response = self._call_s3(
            "upload_part",
            Bucket=self._bucket,
            Key=self._key,
            UploadId=self._upload_id,
            PartNumber=part_number,
            Body=data,
        )
        part: dict[str, Any] = {"PartNumber": part_number, "ETag": response["ETag"]}
        if "ChecksumSHA256" in response:
            part["ChecksumSHA256"] = response["ChecksumSHA256"]
        self._parts[part_number] = part

    def _call_s3(self, method: str, **kwargs: Any) -> Any:
        """Match s3fs' per-file calls, including configured S3 request options."""
        return self._fs.call_s3(
            method, getattr(self._fs, "s3_additional_kwargs", {}), **kwargs
        )

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed S3 multipart writer")


__all__ = ["S3MultipartSeekableWriter", "supports_s3_multipart"]
