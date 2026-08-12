from __future__ import annotations

from collections.abc import Mapping

from refiner.io.datafile import DataFile


def read_blob(reference: Mapping[str, object]) -> bytes:
    """Read the exact byte range described by a Refiner blob reference."""
    path = reference.get("path")
    offset = reference.get("offset")
    size = reference.get("size")
    if not isinstance(path, str) or not path:
        raise ValueError("blob path must be a non-empty string")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("blob offset must be a non-negative integer")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("blob size must be a non-negative integer")

    chunks: list[bytes] = []
    remaining = size
    with DataFile.resolve(path).open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    if remaining:
        received = size - remaining
        raise EOFError(
            f"Blob reference requested {size} bytes at offset {offset}, "
            f"but only {received} bytes were available"
        )
    return b"".join(chunks)


__all__ = ["read_blob"]
