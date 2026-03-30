from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from refiner.io import DataFile
from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePart
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.data.tabular import Tabular, filter_table
from refiner.pipeline.expressions import Expr
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
        filter: Expr | None = None,
        split_row_groups: bool = False,
        file_path_column: str | None = "file_path",
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
            filter: Optional refiner expression used for parquet row-group
                pruning where possible, with residual filtering performed in memory.
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
            file_path_column=file_path_column,
        )
        self.arrow_batch_size = int(arrow_batch_size)
        self.split_row_groups = split_row_groups

        ## filter
        # full filter expression
        self.filter = filter
        # what we can pass down to scanner. essentially an arrow dataset expression
        self._pushdown_filter = (
            filter.extract_pushdown_filter() if filter is not None else None
        )
        # the columns we will need to have in order to be able to apply self.filter
        self._filter_columns = (
            filter.referenced_columns() if filter is not None else set()
        )

        ## columns
        # the columns the parquet reader will return at the end
        self.columns_to_read = (
            tuple(columns_to_read) if columns_to_read is not None else None
        )
        if (
            self.columns_to_read is not None
            and self.file_path_column is not None
            and self.file_path_column in self.columns_to_read
        ):
            raise ValueError(
                "columns_to_read cannot include the synthetic file_path_column; "
                "omit it from columns_to_read and let the reader append it"
            )
        # the columns we will actually load into memory. requested+needed for filtering
        self._scan_columns: list[str] | None = (
            None
            if self.columns_to_read is None
            else sorted(set(self.columns_to_read) | self._filter_columns)
        )

        self._open_pf: Optional[pq.ParquetFile] = None
        self._open_metadata: _ParquetMetadata | None = None
        self._open_fragment_file: DataFile | None = None
        self._open_fragment: ds.ParquetFileFragment | None = None

    def _get_parquet_file(self, source_file: DataFile) -> pq.ParquetFile:
        """Get or open a cached ParquetFile for the current path (single-open-file policy)."""
        fh, opened_new = self._get_file_handle(source_file, mode="rb")
        if opened_new or self._open_pf is None:
            self._open_pf = pq.ParquetFile(fh)
            self._open_metadata = None
        return self._open_pf

    def _get_parquet_fragment(self, source_file: DataFile) -> ds.ParquetFileFragment:
        """Build a Parquet fragment for row-group-aware filter pushdown."""
        if self._open_fragment is not None and self._open_fragment_file == source_file:
            return self._open_fragment

        pyfs = pafs.PyFileSystem(pafs.FSSpecHandler(source_file.fs))
        dataset = ds.dataset(source_file.path, filesystem=pyfs, format="parquet")
        self._open_fragment_file = source_file
        self._open_fragment = next(dataset.get_fragments())
        return self._open_fragment

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "columns_to_read": list(self.columns_to_read)
                if self.columns_to_read is not None
                else None,
                "split_row_groups": self.split_row_groups,
                "filter": self.filter.to_code() if self.filter is not None else None,
            }
        )
        return description

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
            source: DataFile = self.fileset.resolve_file(part.source_index, part.path)
            if part.end == -1:
                yield from self._read_fragment_scan(
                    source,
                    row_groups=self._filtered_row_groups(source, None),
                )
                continue

            pf = self._get_parquet_file(source)
            row_groups, row_bounds = self._row_bounds_for_span(pf, part)
            row_groups = self._filtered_row_groups(source, row_groups)
            if row_groups == []:
                continue
            yield from self._read_fragment_scan(
                source,
                row_groups=row_groups,
                row_bounds=row_bounds,
            )

    def _read_fragment_scan(
        self,
        source_file: DataFile,
        *,
        row_groups: list[int] | None,
        row_bounds: tuple[int, int] | None = None,
    ) -> Iterator[SourceUnit]:
        """Scan the selected parquet row groups and optionally trim to a row interval."""
        row_start = row_end = 0
        if row_bounds is not None:
            row_start, row_end = row_bounds
            if row_groups is None or row_groups == []:
                return

        fragment = self._get_parquet_fragment(source_file)
        if row_groups is not None:
            fragment = fragment.subset(row_group_ids=row_groups)
        if row_bounds is not None:
            assert row_groups is not None
            metadata = self._metadata(self._get_parquet_file(source_file))
            if metadata is None:
                return
            offset = metadata.row_starts[row_groups[0]]
        else:
            offset = 0
        # Row-group pruning already happened before scanning, so scanner only needs
        # projection and batch sizing here.
        for batch in fragment.scanner(
            columns=self._scan_columns,
            filter=None,
            batch_size=self.arrow_batch_size,
        ).to_batches():
            if row_bounds is not None:
                # we need to keep track of start/end of our row bounds
                batch_rows = int(batch.num_rows)
                if offset + batch_rows <= row_start:
                    offset += batch_rows
                    continue
                # Trim the first/last scanned row groups down to the shard's row interval.
                begin = max(0, row_start - offset)
                length = min(batch_rows - begin, row_end - max(row_start, offset))
                if length <= 0:
                    offset += batch_rows
                    if offset >= row_end:
                        return
                    continue
                batch = batch.slice(begin, length)
                offset += batch_rows
            table = pa.Table.from_batches([batch])
            if self.filter is not None:
                table = filter_table(table, self.filter)
            if self.columns_to_read is not None:
                table = table.select(self.columns_to_read)
            table = self._table_with_file_path(table, source_file)
            if table.num_rows > 0:
                yield Tabular(table)
            if row_bounds is not None and offset >= row_end:
                return

    def _filtered_row_groups(
        self,
        source_file: DataFile,
        row_groups: list[int] | None,
    ) -> list[int] | None:
        """Prune row groups via parquet metadata while keeping shard ownership unchanged."""
        if self._pushdown_filter is None:
            return row_groups

        fragment = self._get_parquet_fragment(source_file)
        if row_groups is not None:
            fragment = fragment.subset(row_group_ids=row_groups)
        return [
            row_group.id
            for row_group_fragment in fragment.split_by_row_group(
                filter=self._pushdown_filter
            )
            for row_group in row_group_fragment.row_groups
        ]

    def _row_bounds_for_span(
        self,
        pf: pq.ParquetFile,
        part: FilePart,
    ) -> tuple[list[int] | None, tuple[int, int] | None]:
        """Map a planned byte span to row groups and optional row bounds within those groups."""
        metadata = self._metadata(pf)
        if metadata is None:
            logger.warning(
                "Parquet metadata unavailable for {}; falling back to reading full file for first shard only.",
                part.path,
            )
            if part.start == 0:
                return None, None
            return [], None

        file_size = self.fileset.size(part.source_index, part.path)
        if file_size <= 0:
            return [], None

        if self.split_row_groups:
            if metadata.num_rows <= 0:
                return [], None

            # Row-fraction splitting is an execution-time partitioning heuristic, not a physical byte->row mapping.
            start = max(0, (metadata.num_rows * max(0, part.start)) // file_size)
            end = min(
                metadata.num_rows,
                (metadata.num_rows * min(file_size, part.end)) // file_size,
            )
            if part.end > part.start and end <= start:
                end = min(metadata.num_rows, start + 1)
            starts = metadata.row_starts
            row_bounds = (start, end)
        else:
            starts = metadata.row_group_starts
            start = max(0, part.start)
            end = min(file_size, part.end)
            row_bounds = None

        if not starts or end <= start:
            return [], None

        if row_bounds is not None:
            start_rg = max(0, bisect.bisect_right(starts, start) - 1)
        else:
            # Byte planning stays metadata-free. At read time each row group belongs to the
            # shard whose planned span contains that row group's start offset.
            start_rg = bisect.bisect_left(starts, start)
        end_rg = bisect.bisect_left(starts, end)
        if start_rg >= end_rg:
            if row_bounds is None and start == 0:
                logger.warning(
                    "Parquet row-group byte sizes unavailable for {}; falling back to reading full file for first shard only.",
                    part.path,
                )
                return None, None
            return [], None

        return list(range(start_rg, end_rg)), row_bounds


__all__ = ["ParquetReader"]
