from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from fsspec import AbstractFileSystem
import pyarrow as pa

from refiner.io import DataFile, DataFileSet, DataFolder
from refiner.io.datafile import _file_cache_key
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import DTypeMapping, schema_with_dtypes
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline.data.tabular import repeat_scalar, set_or_append_column
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    BoundedBinaryReader,
    DEFAULT_MAX_AUTOMATIC_SHARDS,
    DEFAULT_TARGET_SHARD_BYTES,
    align_byte_range_to_newlines,
    is_splittable_by_bytes,
    validate_explicit_num_shards,
)
from refiner.worker.context import logger


def _iter_file_part_groups(
    resolved_files: Sequence[tuple[int, DataFile, int]],
    *,
    target_bytes: int,
    split_by_bytes: bool,
    shard_sizes: Sequence[int] | None = None,
) -> Iterator[list[FilePart]]:
    planned_shards = 0
    current_parts: list[FilePart] = []
    current_size = 0

    def can_flush_more() -> bool:
        return shard_sizes is None or planned_shards < len(shard_sizes) - 1

    def next_target_bytes() -> int:
        if shard_sizes is None:
            return target_bytes
        return max(1, shard_sizes[planned_shards])

    for source_index, file, size in resolved_files:
        current_target_bytes = next_target_bytes()
        if not split_by_bytes or not is_splittable_by_bytes(file):
            if (
                current_parts
                and current_size + size > current_target_bytes
                and can_flush_more()
            ):
                yield current_parts
                planned_shards += 1
                current_parts = []
                current_size = 0
            current_parts.append(
                FilePart(
                    path=file.abs_path(),
                    start=0,
                    end=-1,
                    source_index=source_index,
                )
            )
            current_size += size
            continue

        offset = 0
        while offset < size:
            current_target_bytes = next_target_bytes()
            if current_size >= current_target_bytes and can_flush_more():
                yield current_parts
                planned_shards += 1
                current_parts = []
                current_size = 0
                current_target_bytes = next_target_bytes()
            remaining_capacity = current_target_bytes - current_size
            span_size = (
                size - offset
                if remaining_capacity <= 0
                else min(size - offset, remaining_capacity)
            )
            current_parts.append(
                FilePart(
                    path=file.abs_path(),
                    start=offset,
                    end=offset + span_size,
                    source_index=source_index,
                )
            )
            current_size += span_size
            offset += span_size

    if current_parts:
        yield current_parts


def _automatic_plan_within_limit(
    resolved_files: Sequence[tuple[int, DataFile, int]],
    *,
    target_bytes: int,
    split_by_bytes: bool,
    limit: int,
) -> list[list[FilePart]] | None:
    groups = _iter_file_part_groups(
        resolved_files,
        target_bytes=target_bytes,
        split_by_bytes=split_by_bytes,
    )
    planned: list[list[FilePart]] = []
    for group in groups:
        planned.append(group)
        if len(planned) > limit:
            return None
    return planned


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
        include_file: Callable[[str], bool] | None = None,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: str | None = "file_path",
        split_by_bytes: bool = True,
        dtypes: DTypeMapping | None = None,
    ):
        """Create a reader over a set of input files.

        Args:
            inputs: Input spec(s): paths, globs, folders, or `DataFile`/`DataFolder`/`DataFileSet`.
            fs: Optional initialized filesystem to use for string inputs.
            storage_options: Optional fsspec init options (used only when `fs` is not provided).
            recursive: If a directory input is provided, whether to list recursively.
            extensions: If a directory input is provided, filter by these suffixes (case-insensitive).
            include_file: Optional predicate for files discovered from directory inputs.
            target_shard_bytes: Target approximate byte size for planned shards.
            num_shards: Optional explicit number of planned shards.
            dtypes: Optional dtype overrides exposed as this source's schema.
        """
        self.fileset = DataFileSet.resolve(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=extensions,
            include_file=include_file,
        )
        validate_explicit_num_shards(num_shards)
        self.target_shard_bytes = max(1, target_shard_bytes)
        self.num_shards = num_shards
        self.file_path_column = file_path_column
        self.split_by_bytes = split_by_bytes
        self.dtypes = dtypes
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

    @property
    def schema(self) -> pa.Schema | None:
        return schema_with_dtypes(None, getattr(self, "dtypes", None))

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
                inputs.append(entry.abs_path())
        return {
            "path": ", ".join(inputs),
            "inputs": inputs,
            "file_path_column": self.file_path_column,
        }

    def _io_refiner_extras(self) -> tuple[str, ...]:
        return self.fileset.required_refiner_extras()

    def _with_file_path(
        self, row: dict[str, Any], source_file: DataFile
    ) -> dict[str, Any]:
        if self.file_path_column is None or self.file_path_column in row:
            return row
        row[self.file_path_column] = source_file.abs_path()
        return row

    def _table_with_file_path(self, table: pa.Table, source_file: DataFile) -> pa.Table:
        if self.file_path_column is None or self.file_path_column in table.column_names:
            return table
        return set_or_append_column(
            table,
            self.file_path_column,
            repeat_scalar(pa.scalar(source_file.abs_path()), table.num_rows),
        )

    def _get_file_handle(
        self, file: DataFile, *, mode: str = "rb", force_reopen: bool = False
    ):
        """Get a cached file handle for a resolved input file.

        Returns:
            (fh, opened_new): `opened_new` is True if a new file handle was opened.
        """
        if (
            not force_reopen
            and self._open_file is not None
            and _file_cache_key(self._open_file) == _file_cache_key(file)
            and self._open_fh is not None
        ):
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
            - `num_shards` produces exactly that many planned byte buckets when possible;
              otherwise `target_shard_bytes` controls shard size heuristically.
        """
        resolved_files = [
            (source_index, file, self.fileset.size(source_index, file.abs_path()))
            for source_index, files in enumerate(self.fileset.expand_sources())
            for file in files
        ]
        num_shards = self.num_shards
        shard_sizes: list[int] | None = None
        planned_groups: list[list[FilePart]] | None = None
        total_size = sum(size for _, _, size in resolved_files)
        if num_shards is None or num_shards <= 0:
            target_bytes = self.target_shard_bytes
            planned_groups = _automatic_plan_within_limit(
                resolved_files,
                target_bytes=target_bytes,
                split_by_bytes=self.split_by_bytes,
                limit=DEFAULT_MAX_AUTOMATIC_SHARDS,
            )
            if planned_groups is None:
                logger.warning(
                    "{} automatic shard plan exceeds the {}-shard limit; "
                    "increasing shard sizes without dropping input data.",
                    self.name,
                    DEFAULT_MAX_AUTOMATIC_SHARDS,
                )
                num_shards = DEFAULT_MAX_AUTOMATIC_SHARDS

        if num_shards is not None and num_shards > 0:
            # `num_shards` defines exact global byte buckets; the last bucket absorbs any
            # remainder instead of creating an extra shard.
            if total_size <= 0:
                target_bytes = 1
            else:
                base, remainder = divmod(total_size, num_shards)
                # Spread the leftover bytes across the first shards instead of creating
                # one oversized remainder shard at the end.
                shard_sizes = [base + (i < remainder) for i in range(num_shards)]
                target_bytes = max(1, shard_sizes[0])
        groups = (
            planned_groups
            if planned_groups is not None
            else _iter_file_part_groups(
                resolved_files,
                target_bytes=target_bytes,
                split_by_bytes=self.split_by_bytes,
                shard_sizes=shard_sizes,
            )
        )
        shards = [
            Shard.from_file_parts(parts, global_ordinal=ordinal)
            for ordinal, parts in enumerate(groups)
        ]
        if (
            num_shards is not None
            and num_shards > 0
            and not self.split_by_bytes
            and len(resolved_files) < num_shards
        ):
            logger.warning(
                "{} requested {} shards, but this reader keeps files atomic and "
                "only found {} input files; planned {} shards.",
                self.name,
                num_shards,
                len(resolved_files),
                len(shards),
            )
        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read a shard and yield row units.

        Contract:
            - Must accept shards returned by `list_shards()`.
            - Should be safe to call sequentially (single-worker, no concurrent calls).
            - Units can be `Row` or `Tabular`.
        """
        raise NotImplementedError


__all__ = ["BaseReader"]
