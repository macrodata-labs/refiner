from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from typing import Protocol


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> None: ...


@dataclass(frozen=True, slots=True)
class FilePart:
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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilePart:
        return cls(
            path=str(payload["path"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            source_index=int(payload.get("source_index", 0)),
            unit=str(payload.get("unit", "bytes")),
        )

    def update_hash(self, h: _HashWriter) -> None:
        h.update(self.path.encode("utf-8"))
        h.update(b"\0")
        h.update(str(self.start).encode("ascii"))
        h.update(b"\0")
        h.update(str(self.end).encode("ascii"))
        h.update(b"\0")
        h.update(str(self.source_index).encode("ascii"))
        h.update(b"\0")
        h.update(self.unit.encode("utf-8"))
        h.update(b"\0")


@dataclass(frozen=True, slots=True)
class FilePartsDescriptor:
    parts: tuple[FilePart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("shard parts must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"parts": [part.to_dict() for part in self.parts]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilePartsDescriptor:
        parts = payload["parts"]
        if not isinstance(parts, list):
            raise ValueError("file-parts descriptor must contain a parts list")
        return cls(
            tuple(FilePart.from_dict(part) for part in parts if isinstance(part, dict))
        )

    def update_hash(self, h: _HashWriter) -> None:
        h.update(b"file_parts\0")
        for part in self.parts:
            part.update_hash(h)


@dataclass(frozen=True, slots=True)
class Shard:
    descriptor: FilePartsDescriptor  # will later allow other descriptors
    global_ordinal: int | None = None
    start_key: str | None = None
    end_key: str | None = None

    def __post_init__(self) -> None:
        if self.start_key is None:
            object.__setattr__(self, "start_key", self.descriptor.parts[0].locality_key)
        if self.end_key is None:
            object.__setattr__(self, "end_key", self.descriptor.parts[-1].locality_key)

    @classmethod
    def from_file_parts(
        cls,
        file_parts: list[FilePart] | tuple[FilePart, ...],
        *,
        global_ordinal: int | None = None,
        start_key: str | None = None,
        end_key: str | None = None,
    ) -> Shard:
        if not file_parts:
            raise ValueError("file parts must be non-empty")
        return cls(
            descriptor=FilePartsDescriptor(tuple(file_parts)),
            global_ordinal=global_ordinal,
            start_key=start_key,
            end_key=end_key,
        )

    @property
    def id(self) -> str:
        h = hashlib.blake2b(digest_size=6)
        self.descriptor.update_hash(h)
        h.update(str(self.global_ordinal).encode("utf-8"))
        h.update(b"\0")
        h.update((self.start_key or "").encode("utf-8"))
        h.update(b"\0")
        h.update((self.end_key or "").encode("utf-8"))
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.id,
            "global_ordinal": self.global_ordinal,
            "start_key": self.start_key,
            "end_key": self.end_key,
            "descriptor": self.descriptor.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Shard:
        descriptor = payload["descriptor"]
        if not isinstance(descriptor, dict):
            raise ValueError("shard descriptor must be an object")
        global_ordinal = payload.get("global_ordinal")
        start_key = payload.get("start_key")
        end_key = payload.get("end_key")
        return cls(
            descriptor=FilePartsDescriptor.from_dict(descriptor),
            global_ordinal=global_ordinal if isinstance(global_ordinal, int) else None,
            start_key=start_key if isinstance(start_key, str) else None,
            end_key=end_key if isinstance(end_key, str) else None,
        )


def path_hash(path: str, *, source_index: int = 0) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(source_index)).encode("ascii"))
    return h.hexdigest()


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
            Shard.from_file_parts(
                [part for shard in group for part in shard.descriptor.parts],
                global_ordinal=index,
                start_key=group[0].start_key,
                end_key=group[-1].end_key,
            )
        )
    return out


__all__ = [
    "FilePart",
    "FilePartsDescriptor",
    "Shard",
    "coalesce_shards",
    "path_hash",
]
