from __future__ import annotations

import io
from typing import Optional

from fsspec import AbstractFileSystem

DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024


# Extensions that generally imply whole-file/container compression (not safely splittable by byte offsets).
NON_SPLITTABLE_WHOLEFILE_EXTS = (".gz", ".bz2", ".xz", ".zip", ".zst")


def is_splittable_by_bytes(fs: AbstractFileSystem, path: str) -> bool:
    """Return True if the input can be safely sharded by byte offsets."""
    lp = path.lower()
    if lp.endswith(NON_SPLITTABLE_WHOLEFILE_EXTS):
        return False
    # best-effort: if raw file object is not seekable, treat as non-splittable
    try:
        with fs.open(path, mode="rb") as f:
            if hasattr(f, "seekable") and callable(f.seekable):
                return bool(f.seekable())
    except Exception:
        return False
    return True


def align_byte_range_to_newlines(
    fh, *, start: int, end: int, size: int
) -> Optional[tuple[int, int]]:
    """Align a planned byte range to newline boundaries.

    This returns a byte range that can be read as whole lines, using the rule:
      include lines whose start offset s satisfies start <= s < end.

    Returns:
        (aligned_start, aligned_end) or None if the shard is empty.
    """

    def _find_next_nl_pos() -> int:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                return size - 1
            j = chunk.find(b"\n")
            if j != -1:
                return fh.tell() - len(chunk) + j

    if size <= 0 or end <= start:
        return None

    if start <= 0:
        aligned_start = 0
    else:
        fh.seek(start - 1)
        nl = _find_next_nl_pos()
        aligned_start = min(size, nl + 1)

    if end >= size:
        aligned_end = size
    elif end <= 0:
        aligned_end = 0
    else:
        fh.seek(end - 1)
        nl = _find_next_nl_pos()
        aligned_end = min(size, nl + 1)

    if aligned_end <= aligned_start:
        return None
    return aligned_start, aligned_end


class BoundedBinaryReader(io.RawIOBase):
    """A raw binary reader wrapper that stops after `limit` bytes."""

    def __init__(self, raw, limit: int):
        self._raw = raw
        self._remaining = limit

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._remaining <= 0:
            return 0
        max_n = len(b)
        n = max_n if self._remaining >= max_n else self._remaining
        data = self._raw.read(n)
        if not data:
            return 0
        b[: len(data)] = data
        self._remaining -= len(data)
        return len(data)


__all__ = [
    "DEFAULT_TARGET_SHARD_BYTES",
    "NON_SPLITTABLE_WHOLEFILE_EXTS",
    "is_splittable_by_bytes",
    "align_byte_range_to_newlines",
    "BoundedBinaryReader",
]
