from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ShardPart:
    path: str
    start: int
    end: int
    source_index: int = 0
    unit: str = "bytes"

    @property
    def locality_key(self) -> str:
        return path_hash(self.path, source_index=self.source_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": int(self.start),
            "end": int(self.end),
            "source_index": int(self.source_index),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class Shard:
    path: str
    start: int
    end: int
    source_index: int = 0
    unit: str = "bytes"
    parts: tuple[ShardPart, ...] = field(default_factory=tuple)
    global_ordinal: int | None = None
    start_key: str | None = None
    end_key: str | None = None

    def __post_init__(self) -> None:
        if not self.parts:
            parts = (
                ShardPart(
                    path=self.path,
                    start=self.start,
                    end=self.end,
                    source_index=self.source_index,
                    unit=self.unit,
                ),
            )
            object.__setattr__(self, "parts", parts)

        if self.start_key is None:
            object.__setattr__(self, "start_key", self.parts[0].locality_key)
        if self.end_key is None:
            object.__setattr__(self, "end_key", self.parts[-1].locality_key)

    @classmethod
    def from_parts(
        cls,
        parts: list[ShardPart] | tuple[ShardPart, ...],
        *,
        global_ordinal: int | None = None,
        start_key: str | None = None,
        end_key: str | None = None,
    ) -> Shard:
        if not parts:
            raise ValueError("shard parts must be non-empty")
        first = parts[0]
        last = parts[-1]
        return cls(
            path=first.path,
            start=first.start,
            end=last.end,
            source_index=first.source_index,
            unit=first.unit,
            parts=tuple(parts),
            global_ordinal=global_ordinal,
            start_key=start_key,
            end_key=end_key,
        )

    @property
    def id(self) -> str:
        h = hashlib.blake2b(digest_size=6)
        h.update(
            json.dumps(
                {
                    "parts": [part.to_dict() for part in self.parts],
                    "global_ordinal": self.global_ordinal,
                    "start_key": self.start_key,
                    "end_key": self.end_key,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return h.hexdigest()

    def pending_filename(self) -> str:
        return format_pending_filename(shard_id=self.id)

    def leased_filename(self, worker_id: int) -> str:
        return format_leased_filename(shard_id=self.id, worker_id=worker_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.id,
            "path": self.path,
            "start": int(self.start),
            "end": int(self.end),
            "source_index": int(self.source_index),
            "global_ordinal": self.global_ordinal,
            "start_key": self.start_key,
            "end_key": self.end_key,
            "descriptor": {"parts": [part.to_dict() for part in self.parts]},
        }


_RE_SHARD_FILENAME = re.compile(
    r"^(?P<shardid>[0-9a-f]+)(?:__w(?P<workerid>\d+))?\.json$"
)


def path_hash(path: str, *, source_index: int = 0) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(source_index)).encode("ascii"))
    return h.hexdigest()


def format_pending_filename(*, shard_id: str) -> str:
    return f"{shard_id}.json"


def format_leased_filename(*, shard_id: str, worker_id: int) -> str:
    return f"{shard_id}__w{int(worker_id)}.json"


def parse_shard_filename(filename: str) -> tuple[str, int | None]:
    m = _RE_SHARD_FILENAME.match(filename)
    if not m:
        raise ValueError(f"Unrecognized shard filename: {filename!r}")
    worker_id = m.group("workerid")
    return m.group("shardid"), int(worker_id) if worker_id is not None else None


def coalesce_shards(shards: list[Shard], num_shards: int | None) -> list[Shard]:
    if num_shards is None or num_shards <= 0 or num_shards >= len(shards):
        return shards

    out: list[Shard] = []
    total = len(shards)
    for index in range(num_shards):
        start = (index * total) // num_shards
        end = ((index + 1) * total) // num_shards
        group = shards[start:end]
        if not group:
            continue
        out.append(
            Shard.from_parts(
                [part for shard in group for part in shard.parts],
                global_ordinal=index,
                start_key=group[0].start_key,
                end_key=group[-1].end_key,
            )
        )
    return out


__all__ = [
    "ShardPart",
    "Shard",
    "coalesce_shards",
    "path_hash",
    "format_pending_filename",
    "format_leased_filename",
    "parse_shard_filename",
]
