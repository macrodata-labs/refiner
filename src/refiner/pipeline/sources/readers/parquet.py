from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
)


@dataclass(frozen=True, slots=True)
class _ParquetMetadata:
    row_group_starts: tuple[int, ...]
    row_starts: tuple[int, ...]
    num_rows: int


class ParquetReader(BaseReader):
    """Parquet reader planned by byte ranges and resolved through parquet metadata.

    Notes:
        - `list_shards()` only plans byte spans or whole-file atomic shards.
        - `read_shard()` maps planned spans onto row groups by default.
        - `split_row_groups=True` turns planned spans into deterministic row ranges
          inside the relevant row-group window.
    """

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        arrow_batch_size: int = 65536,
        columns_to_read: Sequence[str] | None = None,
        split_row_groups: bool = False,
    ):
        """Create a Parquet reader.

        Args:
            inputs: Input spec(s): paths, globs, folders, or `DataFile`/`DataFolder`/`DataFileSet`.
            fs: Optional initialized filesystem to use for string inputs.
            storage_options: Optional fsspec init options (used only when `fs` is not provided).
            recursive: If a directory input is provided, whether to list recursively.
            target_shard_bytes: Target approximate shard size for planned byte-range shards.
            arrow_batch_size: Max rows per streamed Arrow batch before wrapping as `Tabular`.
            columns_to_read: Optional subset of column names to read (projection pushdown).
            split_row_groups: If True, `read_shard()` can refine planned byte spans into
                deterministic row ranges inside the file. Otherwise reads expand to row groups.

        Notes:
            - `list_shards()` uses file sizes only.
            - `read_shard()` maps planned spans onto row groups or row ranges.
            - Parquet compression (snappy/zstd/etc.) is handled internally by pyarrow.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".parquet", ".pq", ".parq"),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
        )
        self.arrow_batch_size = int(arrow_batch_size)
        self.columns_to_read = (
            tuple(columns_to_read) if columns_to_read is not None else None
        )
        self.split_row_groups = split_row_groups

        self._open_pf: Optional[pq.ParquetFile] = None
        self._open_metadata: _ParquetMetadata | None = None

    def _get_parquet_file(self, source_file) -> pq.ParquetFile:
        """Get or open a cached ParquetFile for the current path (single-open-file policy)."""
        fh, opened_new = self._get_file_handle(source_file, mode="rb")
        if opened_new or self._open_pf is None:
            self._open_pf = pq.ParquetFile(fh)
            self._open_metadata = None
        return self._open_pf

    def _metadata(self, pf: pq.ParquetFile) -> _ParquetMetadata | None:
        """Cache row-group byte starts and row starts for the currently open parquet file."""
        cached = self._open_metadata
        if cached is not None:
            return cached

        md = pf.metadata
        if md is None:
            return None

        row_group_starts: list[int] = []
        row_starts: list[int] = []
        total_bytes = 0
        total_rows = 0
        for index in range(md.num_row_groups):
            row_group_starts.append(total_bytes)
            row_starts.append(total_rows)
            width_raw = md.row_group(index).total_byte_size
            width = int(width_raw) if width_raw is not None else 0
            total_bytes += width
            total_rows += int(md.row_group(index).num_rows)

        self._open_metadata = _ParquetMetadata(
            row_group_starts=tuple(row_group_starts),
            row_starts=tuple(row_starts),
            num_rows=total_rows,
        )
        return self._open_metadata

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one planned parquet shard and yield Arrow-backed tabular units."""
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            pf = self._get_parquet_file(source)
            if part.end == -1:
                for batch in pf.iter_batches(
                    row_groups=None,
                    batch_size=self.arrow_batch_size,
                    columns=self.columns_to_read,
                ):
                    yield Tabular.from_batch(batch)
                continue

            if self.split_row_groups:
                # For explicit intra-file splitting we treat the planned byte span as a deterministic row fraction.
                yield from self._read_row_fraction(pf, part)
                continue

            # Default parquet behavior expands the planned byte span to the covering row groups.
            rg_indices = self._row_groups_for_span(pf, part)
            if rg_indices == []:
                continue
            for batch in pf.iter_batches(
                row_groups=rg_indices,
                batch_size=self.arrow_batch_size,
                columns=self.columns_to_read,
            ):
                yield Tabular.from_batch(batch)

    def _row_groups_for_span(self, pf: pq.ParquetFile, part) -> list[int] | None:
        """Map a planned byte span to the row groups whose starts fall inside that span."""
        metadata = self._metadata(pf)
        if metadata is None:
            if part.start == 0:
                logger.warning(
                    "Parquet metadata unavailable for {}; falling back to reading full file for first shard only.",
                    part.path,
                )
                return None
            return []

        row_group_starts = metadata.row_group_starts
        if not row_group_starts:
            return []

        file_size = self.fileset.size(part.source_index, part.path)
        if file_size <= 0:
            return []

        start = max(0, part.start)
        end = min(file_size, part.end)
        if end <= start:
            return []

        # Byte planning stays metadata-free. At read time each row group belongs to the
        # shard whose planned span contains that row group's start offset.
        start_rg = bisect.bisect_left(row_group_starts, start)
        end_rg = bisect.bisect_left(row_group_starts, end)
        if start_rg >= end_rg:
            if start == 0:
                logger.warning(
                    "Parquet row-group byte sizes unavailable for {}; falling back to reading full file for first shard only.",
                    part.path,
                )
                return None
            return []

        return list(range(start_rg, end_rg))

    def _read_row_fraction(self, pf: pq.ParquetFile, part) -> Iterator[SourceUnit]:
        """Read a planned parquet span as a deterministic row fraction within relevant row groups."""
        metadata = self._metadata(pf)
        if metadata is None:
            for batch in pf.iter_batches(
                row_groups=None,
                batch_size=self.arrow_batch_size,
                columns=self.columns_to_read,
            ):
                yield Tabular.from_batch(batch)
            return

        file_size = self.fileset.size(part.source_index, part.path)
        if metadata.num_rows <= 0 or file_size <= 0:
            return

        # Row-fraction splitting is an execution-time partitioning heuristic, not a physical byte->row mapping.
        row_start = max(0, (metadata.num_rows * max(0, part.start)) // file_size)
        row_end = (
            metadata.num_rows
            if part.end == -1
            else min(
                metadata.num_rows,
                (metadata.num_rows * min(file_size, part.end)) // file_size,
            )
        )
        if part.end > part.start and row_end <= row_start:
            row_end = min(metadata.num_rows, row_start + 1)
        if row_end <= row_start:
            return

        start_rg = bisect.bisect_right(metadata.row_starts, row_start) - 1
        end_rg = bisect.bisect_left(metadata.row_starts, row_end)
        start_rg = max(0, start_rg)
        row_groups = list(range(start_rg, end_rg)) if start_rg < end_rg else None
        offset = metadata.row_starts[start_rg] if row_groups else 0
        remaining = row_end - row_start
        for batch in pf.iter_batches(
            row_groups=row_groups,
            batch_size=self.arrow_batch_size,
            columns=self.columns_to_read,
        ):
            batch_rows = int(batch.num_rows)
            if offset + batch_rows <= row_start:
                offset += batch_rows
                continue
            begin = max(0, row_start - offset)
            length = min(batch_rows - begin, remaining)
            if length > 0:
                yield Tabular.from_batch(batch.slice(begin, length))
                remaining -= length
            offset += batch_rows
            if remaining <= 0:
                break


__all__ = ["ParquetReader"]
