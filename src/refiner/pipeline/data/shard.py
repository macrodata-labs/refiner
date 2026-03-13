from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Shard:
    """A unit of read work.

    Notes:
        - `path` is stored in the form expected by fsspec (`fs.open/fs.exists`).
        - `start/end` are numeric offsets interpreted by the reader (e.g. bytes or row-group indices).
    """

    path: str
    start: int
    end: int

    @property
    def id(self) -> str:
        """Stable identifier for this shard."""
        # 12 hex chars = 48 bits; short for filenames/keys with negligible collision risk for typical workloads.
        h = hashlib.blake2b(digest_size=6)
        h.update(self.path.encode("utf-8"))
        h.update(b"\0")
        h.update(str(int(self.start)).encode("ascii"))
        h.update(b"\0")
        h.update(str(int(self.end)).encode("ascii"))
        return h.hexdigest()

    @property
    def file_key(self) -> str:
        """Stable per-file key used for locality/blocks."""
        return path_hash(self.path)

    def pending_filename(self) -> str:
        return format_pending_filename(
            pathhash=self.file_key, start=self.start, end=self.end, shard_id=self.id
        )

    def leased_filename(self, worker_id: int) -> str:
        return format_leased_filename(
            pathhash=self.file_key,
            start=self.start,
            end=self.end,
            shard_id=self.id,
            worker_id=worker_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.id,
            "path": self.path,
            "start": int(self.start),
            "end": int(self.end),
        }


_RE_SHARD_FILENAME = re.compile(
    r"^(?P<pathhash>[0-9a-f]+)__"
    r"(?P<start>\d{20})__"
    r"(?P<end>\d{20})__"
    r"(?P<shardid>[0-9a-f]+)"
    r"(?:__w(?P<workerid>\d+))?"
    r"\.json$"
)


def path_hash(path: str) -> str:
    # 12 hex chars, matches Shard.id length.
    h = hashlib.blake2b(digest_size=6)
    h.update(path.encode("utf-8"))
    return h.hexdigest()


def format_pending_filename(
    *, pathhash: str, start: int, end: int, shard_id: str
) -> str:
    return f"{pathhash}__{int(start):020d}__{int(end):020d}__{shard_id}.json"


def format_leased_filename(
    *, pathhash: str, start: int, end: int, shard_id: str, worker_id: int
) -> str:
    base = format_pending_filename(
        pathhash=pathhash, start=start, end=end, shard_id=shard_id
    )[:-5]  # strip .json
    return f"{base}__w{int(worker_id)}.json"


def parse_shard_filename(filename: str) -> tuple[str, int, int, str, int | None]:
    """Parse pending/leased/done/failed shard filename.

    Returns: (pathhash, start, end, shard_id, worker_id)
    """
    m = _RE_SHARD_FILENAME.match(filename)
    if not m:
        raise ValueError(f"Unrecognized shard filename: {filename!r}")
    pathhash = m.group("pathhash")
    start = int(m.group("start"))
    end = int(m.group("end"))
    shard_id = m.group("shardid")
    w = m.group("workerid")
    worker_id = int(w) if w is not None else None
    return pathhash, start, end, shard_id, worker_id


def strip_worker_suffix(filename: str) -> str:
    """Convert a leased filename to its pending/done/failed basename."""
    pathhash, start, end, shard_id, _ = parse_shard_filename(filename)
    return format_pending_filename(
        pathhash=pathhash, start=start, end=end, shard_id=shard_id
    )


__all__ = [
    "Shard",
    "path_hash",
    "format_pending_filename",
    "format_leased_filename",
    "parse_shard_filename",
    "strip_worker_suffix",
]
