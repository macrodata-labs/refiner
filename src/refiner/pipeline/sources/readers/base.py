from __future__ import annotations

import io
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem

from refiner.io import DataFile, DataFileSet, DataFolder
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    BoundedBinaryReader,
    DEFAULT_TARGET_SHARD_BYTES,
    align_byte_range_to_newlines,
    is_splittable_by_bytes,
)


class BaseReader(BaseSource):
    """Base class for file-backed readers.

    Responsibilities:
        - Normalize input sources without eagerly listing them.
        - Lazily expand those sources into a deterministic list of concrete input files.
        - Plan file-backed shards as ordered byte spans across all resolved files.
        - Leave final boundary decisions to concrete readers at `read_shard()` time.

    Note:
        This object is expected to be used by a single worker at a time (no concurrent read_shard calls).
    """

    name: str = ""

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        extensions: Sequence[str] = (),
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
    ):
        """Create a reader over a set of input files.

        Args:
            inputs: Input spec(s): paths, globs, folders, or `DataFile`/`DataFolder`/`DataFileSet`.
            fs: Optional initialized filesystem to use for string inputs.
            storage_options: Optional fsspec init options (used only when `fs` is not provided).
            recursive: If a directory input is provided, whether to list recursively.
            extensions: If a directory input is provided, filter by these suffixes (case-insensitive).
            target_shard_bytes: Target approximate byte size for planned shards.
            num_shards: Optional explicit number of planned shards.
        """
        self.fileset = DataFileSet.resolve(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=extensions,
        )
        self.target_shard_bytes = max(1, target_shard_bytes)
        self.num_shards = num_shards if num_shards is not None else None
        self.split_by_bytes = True
        # Single-open-file cache for readers that do byte-based seeks or stream reads.
        self._open_file: DataFile | None = None
        self._open_fh: Any | None = None

        if not self.name:
            reader_name = self.__class__.__name__.replace("Reader", "").lower()
            self.name = f"read_{reader_name}"

    @property
    def files(self) -> list[str]:
        """Deterministic list of resolved input file paths."""
        return [file.path for file in self.fileset.files]

    def describe(self) -> dict[str, Any]:
        # Keep planning metadata cheap: do not resolve/list inputs here.
        entries = self.fileset.entries
        if not entries:
            return {}
        inputs: list[str] = []
        for entry in entries:
            if isinstance(entry, DataFile):
                inputs.append(entry.abs_path())
            elif isinstance(entry, Path):
                inputs.append(str(entry))
            elif isinstance(entry, DataFolder):
                inputs.append(str(entry.abs_paths("")))
            else:
                inputs.append(str(entry.fs.unstrip_protocol(entry.path)))
        return {"path": ", ".join(inputs), "inputs": inputs}

    def _get_file_handle(
        self, file: DataFile, *, mode: str = "rb", force_reopen: bool = False
    ):
        """Get a cached file handle for a resolved input file.

        Returns:
            (fh, opened_new): `opened_new` is True if a new file handle was opened.
        """
        if not force_reopen and self._open_file == file and self._open_fh is not None:
            return self._open_fh, False

        if self._open_fh is not None:
            try:
                self._open_fh.close()
            except Exception:
                pass
            self._open_fh = None
            self._open_file = None

        self._open_fh = file.open(mode=mode)
        self._open_file = file
        return self._open_fh, True

    def _open_aligned_byte_span(
        self, part: FilePart
    ) -> tuple[DataFile, io.BufferedReader, int] | None:
        """Open a planned byte span after snapping it to newline boundaries.

        This is shared by line-oriented readers such as JSONL and CSV. Parquet does
        not use it because parquet translates planned byte spans through metadata
        instead of reading raw file bytes directly.
        """
        source = self.fileset.resolve_file(part.source_index, part.path)
        fh, _ = self._get_file_handle(source, mode="rb")
        size = self.fileset.size(part.source_index, part.path)
        aligned = align_byte_range_to_newlines(
            fh, start=part.start, end=part.end, size=size
        )
        if aligned is None:
            return None
        start, end = aligned

        try:
            fh.seek(start)
        except Exception:
            fh, _ = self._get_file_handle(source, mode="rb", force_reopen=True)
            fh.seek(start)

        return source, io.BufferedReader(BoundedBinaryReader(fh, end - start)), start

    def list_shards(self) -> list[Shard]:
        """Return the deterministic list of shards for this reader.

        Contract:
            - Shards must only reference resolved input files.
            - File readers plan shards as byte/file spans only.
            - Each shard part carries the source entry index it belongs to.
            - Read-time boundary adaptation is handled by each concrete reader.

        Notes:
            - Splittable files contribute raw byte spans.
            - Atomic files stay whole with `start=0, end=-1`.
            - `num_shards` partitions total planned bytes; otherwise `target_shard_bytes`
              controls shard size heuristically.
        """
        num_shards = self.num_shards
        if num_shards is None or num_shards <= 0:
            target_bytes = self.target_shard_bytes
        else:
            # `num_shards` partitions the total planned byte span across all resolved files.
            total_size = 0
            for source_index, files in enumerate(self.fileset.expand_sources()):
                for file in files:
                    total_size += self.fileset.size(source_index, file.abs_path())
            target_bytes = 1 if total_size <= 0 else max(1, total_size // num_shards)
        shards: list[Shard] = []
        current_parts: list[FilePart] = []
        current_size = 0

        def flush() -> None:
            nonlocal current_parts, current_size
            if not current_parts:
                return
            shards.append(
                Shard.from_file_parts(current_parts, global_ordinal=len(shards))
            )
            current_parts = []
            current_size = 0

        for source_index, files in enumerate(self.fileset.expand_sources()):
            for file in files:
                path = file.abs_path()
                size = self.fileset.size(source_index, path)
                # Atomic files stay whole: `end=-1` means "reader decides how to read the full file".
                if not self.split_by_bytes or not is_splittable_by_bytes(
                    file.fs, file.path
                ):
                    if current_parts and current_size + size > target_bytes:
                        flush()
                    current_parts.append(
                        FilePart(
                            path=path,
                            start=0,
                            end=-1,
                            source_index=source_index,
                        )
                    )
                    current_size += size
                    continue

                offset = 0
                while offset < size:
                    # Splittable files are planned as raw byte spans; readers snap these to real boundaries later.
                    if current_size >= target_bytes:
                        flush()
                    span_size = min(size - offset, target_bytes - current_size)
                    current_parts.append(
                        FilePart(
                            path=path,
                            start=offset,
                            end=offset + span_size,
                            source_index=source_index,
                        )
                    )
                    current_size += span_size
                    offset += span_size

        flush()
        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read a shard and yield row units.

        Contract:
            - Must accept shards returned by `list_shards()`.
            - Should be safe to call sequentially (single-worker, no concurrent calls).
            - Units can be `Row` or Arrow tabular blocks (`RecordBatch`/`Table`).
        """
        raise NotImplementedError


__all__ = ["BaseReader"]
