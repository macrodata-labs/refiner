from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import threading
from typing import Any


_MIN_PART_BYTES = 5 << 20
_MAX_PART_BYTES = 5 << 30
_MAX_PARTS = 10_000
DEFAULT_PART_BYTES = 64 << 20
DEFAULT_MAX_IN_FLIGHT = 4

_PATCH_MARKER = "_refiner_concurrent_multipart"
_PART_BYTES_ATTRIBUTE = "_refiner_multipart_part_bytes"
_MAX_IN_FLIGHT_ATTRIBUTE = "_refiner_multipart_max_in_flight"
_INSTALL_LOCK = threading.Lock()

ConcurrentS3File: type[Any] | None = None


def is_s3fs(filesystem: object) -> bool:
    """Return whether *filesystem* is an s3fs filesystem without importing s3fs."""
    return any(
        cls.__module__.split(".", maxsplit=1)[0] == "s3fs"
        for cls in type(filesystem).__mro__
    )


def part_bytes_for_size(
    size_hint: int,
    *,
    minimum: int = DEFAULT_PART_BYTES,
) -> int:
    """Choose a fixed part size that stays within S3's object limits."""
    if minimum < _MIN_PART_BYTES:
        raise ValueError("S3 multipart parts must be at least 5 MiB")
    if size_hint > 5 << 40:
        raise ValueError("S3 blob size exceeds the 5 TiB multipart object limit")
    part_bytes = max(minimum, (size_hint + _MAX_PARTS - 1) // _MAX_PARTS)
    if part_bytes > _MAX_PART_BYTES:
        raise ValueError("S3 blob size exceeds the 5 TiB multipart object limit")
    return part_bytes


def _part_header(part_number: int, response: dict[str, Any]) -> dict[str, Any]:
    part = {"PartNumber": part_number, "ETag": response["ETag"]}
    if checksum := response.get("ChecksumSHA256"):
        part["ChecksumSHA256"] = checksum
    return part


def _upload_batch(file: Any, chunks: list[bytes]) -> list[dict[str, Any]]:
    first_part_number = len(file.parts) + 1
    if first_part_number + len(chunks) - 1 > _MAX_PARTS:
        raise ValueError(
            "S3 multipart upload exceeded 10,000 parts; provide a final size hint"
        )

    def upload(item: tuple[int, bytes]) -> dict[str, Any]:
        part_number, payload = item
        response = file._call_s3(
            "upload_part",
            Bucket=file.bucket,
            PartNumber=part_number,
            UploadId=file.mpu["UploadId"],
            Body=payload,
            Key=file.key,
        )
        return _part_header(part_number, response)

    numbered_chunks = list(enumerate(chunks, start=first_part_number))
    with ThreadPoolExecutor(
        max_workers=len(numbered_chunks),
        thread_name_prefix="refiner-s3-part",
    ) as executor:
        return list(executor.map(upload, numbered_chunks))


def _discard_after_failure(file: Any) -> None:
    try:
        file.discard()
    except Exception:
        # The original write/commit failure is more useful than an abort failure.
        file.mpu = None
        file.buffer = None
    finally:
        file.closed = True


def _concurrent_upload_chunk(file: Any, *, final: bool) -> bool:
    if (
        file.autocommit
        and not file.append_block
        and final
        and file.tell() < file.blocksize
    ):
        return file._refiner_original_upload_chunk(final=final)

    try:
        file.buffer.seek(0)
        while True:
            bytes_left = len(file.buffer.getbuffer()) - file.buffer.tell()
            if not final and bytes_left < file._refiner_batch_bytes:
                break
            if final and bytes_left == 0:
                break

            count = min(
                file._refiner_max_in_flight,
                (bytes_left + file._refiner_part_bytes - 1) // file._refiner_part_bytes,
            )
            chunks = [file.buffer.read(file._refiner_part_bytes) for _ in range(count)]
            file.parts.extend(_upload_batch(file, chunks))

        file.offset += file.buffer.tell()
        file.buffer = io.BytesIO(file.buffer.read())
        file.buffer.seek(0, 2)

        if file.autocommit and final:
            file.commit()
    except BaseException:
        _discard_after_failure(file)
        raise
    return False


def _build_concurrent_s3_file(original: type[Any]) -> type[Any]:
    class _ConcurrentS3File(original):
        _refiner_concurrent_multipart = True
        _refiner_original_upload_chunk = original._upload_chunk

        def __init__(
            self,
            s3: Any,
            path: str,
            mode: str = "rb",
            *args: Any,
            **kwargs: Any,
        ) -> None:
            if "r" not in mode:
                requested_block_size = args[0] if args else kwargs.get("block_size")
                default_block_size = getattr(s3, "default_block_size", None)
                requested_part_bytes = (
                    requested_block_size
                    if requested_block_size not in (None, default_block_size)
                    else DEFAULT_PART_BYTES
                )
                configured_part_bytes = getattr(
                    s3,
                    _PART_BYTES_ATTRIBUTE,
                    requested_part_bytes,
                )
                configured_max_in_flight = getattr(
                    s3,
                    _MAX_IN_FLIGHT_ATTRIBUTE,
                    min(
                        DEFAULT_MAX_IN_FLIGHT,
                        getattr(s3, "max_concurrency", DEFAULT_MAX_IN_FLIGHT),
                    ),
                )
                if configured_max_in_flight <= 0:
                    raise ValueError("S3 multipart max_in_flight must be > 0")
                self._refiner_part_bytes = part_bytes_for_size(
                    kwargs.get("size") or 0,
                    minimum=configured_part_bytes,
                )
                self._refiner_max_in_flight = configured_max_in_flight
                self._refiner_batch_bytes = (
                    self._refiner_part_bytes * self._refiner_max_in_flight
                )
                if args:
                    args = (self._refiner_part_bytes, *args[1:])
                else:
                    kwargs["block_size"] = self._refiner_part_bytes

            super().__init__(s3, path, mode, *args, **kwargs)

        def _upload_chunk(self, final: bool = False) -> bool:
            return _concurrent_upload_chunk(self, final=final)

    _ConcurrentS3File.__name__ = "ConcurrentS3File"
    _ConcurrentS3File.__qualname__ = "ConcurrentS3File"
    _ConcurrentS3File.__module__ = __name__
    return _ConcurrentS3File


def install_s3fs_concurrent_writes() -> bool:
    """Replace s3fs's file class with Refiner's concurrent multipart subclass."""
    global ConcurrentS3File

    try:
        core = importlib.import_module("s3fs.core")
        s3fs = importlib.import_module("s3fs")
    except ModuleNotFoundError as error:
        if error.name not in {"s3fs", "s3fs.core"}:
            raise
        return False

    with _INSTALL_LOCK:
        current = core.S3File
        if getattr(current, _PATCH_MARKER, False):
            ConcurrentS3File = current
            s3fs.S3File = current
            return True

        required_methods = ("_upload_chunk", "commit", "discard")
        if not all(hasattr(current, method) for method in required_methods):
            return False

        patched = _build_concurrent_s3_file(current)
        ConcurrentS3File = patched
        core.S3File = patched
        s3fs.S3File = patched
        return True


def install_s3fs_concurrent_writes_for(filesystem: object) -> bool:
    """Install the process-wide patch when *filesystem* is backed by s3fs."""
    return is_s3fs(filesystem) and install_s3fs_concurrent_writes()


__all__ = [
    "ConcurrentS3File",
    "DEFAULT_MAX_IN_FLIGHT",
    "DEFAULT_PART_BYTES",
    "install_s3fs_concurrent_writes",
    "install_s3fs_concurrent_writes_for",
    "is_s3fs",
    "part_bytes_for_size",
]
