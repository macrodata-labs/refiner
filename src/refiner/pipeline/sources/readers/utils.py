from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from typing import cast

from fsspec import AbstractFileSystem

DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024
PathSelection = Mapping[str, str] | Sequence[str] | str


# Extensions that generally imply whole-file/container compression (not safely splittable by byte offsets).
NON_SPLITTABLE_WHOLEFILE_EXTS = (".gz", ".bz2", ".xz", ".zip", ".zst")


def decode_value(
    value: Any,
    *,
    decode_bytes: bool = True,
    preserve_arrays: bool = False,
) -> Any:
    if isinstance(value, bytes):
        if not decode_bytes:
            return value
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, str) and any("\udc80" <= char <= "\udcff" for char in value):
        return value.encode("utf-8", errors="surrogateescape")
    if hasattr(value, "shape") and value.shape == ():
        return decode_value(
            value.item(),
            decode_bytes=decode_bytes,
            preserve_arrays=preserve_arrays,
        )
    if hasattr(value, "tolist"):
        if preserve_arrays and getattr(
            getattr(value, "dtype", None), "kind", None
        ) not in (
            "O",
            "S",
        ):
            return value
        return decode_value(
            value.tolist(),
            decode_bytes=decode_bytes,
            preserve_arrays=preserve_arrays,
        )
    if isinstance(value, list):
        return [
            decode_value(
                item,
                decode_bytes=decode_bytes,
                preserve_arrays=preserve_arrays,
            )
            for item in value
        ]
    return value


def path_selection_map(
    value: PathSelection | None,
    *,
    format_name: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {value.rsplit("/", 1)[-1]: value}
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, str], value))
    out: dict[str, str] = {}
    for path in value:
        name = path.rsplit("/", 1)[-1]
        if name in out:
            raise ValueError(
                f"{format_name} path selections must have unique derived column names; "
                f"use an explicit mapping for duplicate name {name!r}"
            )
        out[name] = path
    return out


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
    "PathSelection",
    "decode_value",
    "path_selection_map",
    "is_splittable_by_bytes",
    "align_byte_range_to_newlines",
    "BoundedBinaryReader",
]
