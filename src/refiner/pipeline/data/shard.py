from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> None: ...


@dataclass(frozen=True, slots=True)
class FilePart:
    """One planned file span inside a file-backed shard descriptor.

    `start/end` are planning offsets only:
        - `end == -1` means the file is atomic and should be read whole.
        - Otherwise readers adapt the span to the boundary that makes sense for
          that format at `read_shard()` time.
    """

    path: str
    start: int
    end: int
    source_index: int = 0

    @property
    def locality_key(self) -> str:
        return path_hash(self.path, source_index=self.source_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": int(self.start),
            "end": int(self.end),
            "source_index": int(self.source_index),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilePart:
        return cls(
            path=str(payload["path"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            source_index=int(payload.get("source_index", 0)),
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


@dataclass(frozen=True, slots=True)
class FilePartsDescriptor:
    """Descriptor for file-backed shards planned as one or more file spans."""

    parts: tuple[FilePart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("shard parts must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "file_parts", "parts": [part.to_dict() for part in self.parts]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilePartsDescriptor:
        parts = payload["parts"]
        if not isinstance(parts, list):
            raise ValueError("file-parts descriptor must contain a parts list")
        if any(not isinstance(part, dict) for part in parts):
            raise ValueError("file-parts descriptor parts must be objects")
        return cls(tuple(FilePart.from_dict(part) for part in parts))

    def update_hash(self, h: _HashWriter) -> None:
        h.update(b"file_parts\0")
        for part in self.parts:
            part.update_hash(h)

    @property
    def descriptor_start_key(self) -> str:
        return self.parts[0].locality_key

    @property
    def descriptor_end_key(self) -> str:
        return self.parts[-1].locality_key


@dataclass(frozen=True, slots=True)
class RowRangeDescriptor:
    """Descriptor for synthetic row-backed sources such as `from_items()`."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("row-range end must be >= start")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "row_range",
            "start": int(self.start),
            "end": int(self.end),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RowRangeDescriptor:
        return cls(
            start=int(payload["start"]),
            end=int(payload["end"]),
        )

    def update_hash(self, h: _HashWriter) -> None:
        h.update(b"row_range\0")
        h.update(str(self.start).encode("ascii"))
        h.update(b"\0")
        h.update(str(self.end).encode("ascii"))
        h.update(b"\0")

    @property
    def descriptor_start_key(self) -> str | None:
        return None

    @property
    def descriptor_end_key(self) -> str | None:
        return None


ShardDescriptor = FilePartsDescriptor | RowRangeDescriptor


@dataclass(frozen=True, slots=True)
class Shard:
    """Logical unit of source work plus scheduling hints.

    The descriptor says what to read. `global_ordinal`, `start_key`, and
    `end_key` are planning hints used by the local/cloud claim logic.
    """

    descriptor: ShardDescriptor
    global_ordinal: int | None = None
    start_key: str | None = None
    end_key: str | None = None

    def __post_init__(self) -> None:
        if self.start_key is None:
            object.__setattr__(self, "start_key", self.descriptor.descriptor_start_key)
        if self.end_key is None:
            object.__setattr__(self, "end_key", self.descriptor.descriptor_end_key)

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

    @classmethod
    def from_row_range(
        cls,
        *,
        start: int,
        end: int,
        global_ordinal: int | None = None,
        start_key: str | None = None,
        end_key: str | None = None,
    ) -> Shard:
        return cls(
            descriptor=RowRangeDescriptor(
                start=start,
                end=end,
            ),
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
        kind = descriptor.get("kind")
        if kind == "file_parts":
            parsed_descriptor: ShardDescriptor = FilePartsDescriptor.from_dict(
                descriptor
            )
        elif kind == "row_range":
            parsed_descriptor = RowRangeDescriptor.from_dict(descriptor)
        else:
            raise ValueError(f"unsupported shard descriptor kind: {kind!r}")
        global_ordinal = payload.get("global_ordinal")
        start_key = payload.get("start_key")
        end_key = payload.get("end_key")
        if global_ordinal is not None and not isinstance(global_ordinal, int):
            raise ValueError("global_ordinal must be an integer or null")
        if start_key is not None and not isinstance(start_key, str):
            raise ValueError("start_key must be a string or null")
        if end_key is not None and not isinstance(end_key, str):
            raise ValueError("end_key must be a string or null")
        return cls(
            descriptor=parsed_descriptor,
            global_ordinal=global_ordinal,
            start_key=start_key,
            end_key=end_key,
        )


def path_hash(path: str, *, source_index: int = 0) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(source_index)).encode("ascii"))
    return h.hexdigest()


__all__ = [
    "FilePart",
    "FilePartsDescriptor",
    "RowRangeDescriptor",
    "Shard",
    "ShardDescriptor",
    "path_hash",
]
