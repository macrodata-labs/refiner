from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Optional

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePart
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    clamp_target_bytes,
)


class ParquetReader(BaseReader):
    """Parquet reader sharded by row groups (per-file)."""

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        arrow_batch_size: int = 65536,
        columns_to_read: Sequence[str] | None = None,
        sharding_mode: Literal["rowgroups", "bytes_lazy"] = "rowgroups",
        split_row_groups: bool = False,
    ):
        """Create a Parquet reader.

        Args:
            inputs: Input spec(s): paths, globs, folders, or `DataFile`/`DataFolder`/`DataFileSet`.
            fs: Optional initialized filesystem to use for string inputs.
            storage_options: Optional fsspec init options (used only when `fs` is not provided).
            recursive: If a directory input is provided, whether to list recursively.
            target_shard_bytes: Target approximate shard size (row groups are grouped up to this size).
            arrow_batch_size: Max rows per Arrow `RecordBatch` yielded while reading.
            columns_to_read: Optional subset of column names to read (projection pushdown).
            sharding_mode: Either:
                - `"rowgroups"`: list_shards reads parquet metadata and returns row-group shards (more exact).
                - `"bytes_lazy"`: list_shards uses file sizes only; `read_shard` maps planned byte-ranges to row groups
                  after loading metadata. This can be faster when there are many input files.

        Notes:
            - Shards are contiguous row-group ranges within each file.
            - Parquet compression (snappy/zstd/etc.) is handled internally by pyarrow.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".parquet", ".pq", ".parq"),
        )
        self.target_shard_bytes = clamp_target_bytes(target_shard_bytes)
        self.arrow_batch_size = int(arrow_batch_size)
        self.columns_to_read = (
            tuple(columns_to_read) if columns_to_read is not None else None
        )
        if sharding_mode not in ("rowgroups", "bytes_lazy"):
            raise ValueError("sharding_mode must be 'rowgroups' or 'bytes_lazy'")
        self.sharding_mode = sharding_mode
        self.split_row_groups = split_row_groups

        self._open_pf: Optional[pq.ParquetFile] = None

    def _get_parquet_file(self, source_file) -> pq.ParquetFile:
        """Get or open a cached ParquetFile for the current path (single-open-file policy)."""
        fh, opened_new = self._get_file_handle(source_file, mode="rb")
        if opened_new or self._open_pf is None:
            # fsspec file objects are usually seekable and acceptable for pyarrow.parquet
            self._open_pf = pq.ParquetFile(fh)
        return self._open_pf

    def list_shards(self) -> list[Shard]:
        """List shards for all resolved input files.

        For Parquet:
            - If `sharding_mode == "rowgroups"`:
                - file-part `start` / `end` are **row-group indices**.
                - Row-group ranges are half-open: `[start, end)`.
            - If `sharding_mode == "bytes_lazy"`:
                - file-part `start` / `end` are **byte offsets** for planning only.
                - `read_shard()` will load metadata and map the byte-range deterministically onto row groups.
            - `end == -1` is used as a sentinel meaning \"read the whole file\".
        """
        shards: list[Shard] = []
        global_ordinal = 0
        for source_index, files in enumerate(self.fileset.expand_sources()):
            for file in files:
                path = file.abs_path()
                size = self.fileset.size(source_index, path)

                if size <= self.target_shard_bytes and (
                    self.sharding_mode != "rowgroups" or not self.split_row_groups
                ):
                    shards.append(
                        Shard.from_file_parts(
                            [
                                FilePart(
                                    path=path,
                                    start=0,
                                    end=-1,
                                    source_index=source_index,
                                )
                            ],
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1
                    continue

                if self.sharding_mode == "bytes_lazy":
                    start = 0
                    while start < size:
                        end = min(size, start + self.target_shard_bytes)
                        shards.append(
                            Shard.from_file_parts(
                                [
                                    FilePart(
                                        path=path,
                                        start=start,
                                        end=end,
                                        source_index=source_index,
                                    )
                                ],
                                global_ordinal=global_ordinal,
                            )
                        )
                        global_ordinal += 1
                        start = end
                    continue

                pf = self._get_parquet_file(file)
                md = pf.metadata
                if md is None:
                    shards.append(
                        Shard.from_file_parts(
                            [
                                FilePart(
                                    path=path,
                                    start=0,
                                    end=-1,
                                    source_index=source_index,
                                )
                            ],
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1
                    continue

                n = md.num_row_groups
                if n <= 0:
                    continue

                start = 0
                acc = 0
                for rg in range(n):
                    rg_bytes = md.row_group(rg).total_byte_size
                    rg_bytes = int(rg_bytes) if rg_bytes is not None else 0
                    rg_rows = md.row_group(rg).num_rows
                    rg_rows = int(rg_rows) if rg_rows is not None else 0

                    if (
                        self.split_row_groups
                        and rg == start
                        and rg_rows > 1
                        and (rg_bytes <= 0 or rg_bytes > self.target_shard_bytes)
                    ):
                        rows_per_shard = max(
                            1,
                            int(rg_rows * (self.target_shard_bytes / max(1, rg_bytes))),
                        )
                        row_start = 0
                        while row_start < rg_rows:
                            row_end = min(rg_rows, row_start + rows_per_shard)
                            shards.append(
                                Shard.from_file_parts(
                                    [
                                        FilePart(
                                            path=path,
                                            start=row_start,
                                            end=row_end,
                                            source_index=source_index,
                                            unit=f"rows:{rg}",
                                        )
                                    ],
                                    global_ordinal=global_ordinal,
                                )
                            )
                            global_ordinal += 1
                            row_start = row_end
                        start = rg + 1
                        acc = 0
                        continue

                    if rg == start:
                        acc = rg_bytes
                        continue

                    if acc >= self.target_shard_bytes:
                        end = rg
                        shards.append(
                            Shard.from_file_parts(
                                [
                                    FilePart(
                                        path=path,
                                        start=start,
                                        end=end,
                                        source_index=source_index,
                                        unit="row_groups",
                                    )
                                ],
                                global_ordinal=global_ordinal,
                            )
                        )
                        global_ordinal += 1
                        start = rg
                        acc = rg_bytes
                    else:
                        acc += rg_bytes

                if start < n:
                    shards.append(
                        Shard.from_file_parts(
                            [
                                FilePart(
                                    path=path,
                                    start=start,
                                    end=n,
                                    source_index=source_index,
                                    unit="row_groups",
                                )
                            ],
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1

        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read a parquet shard and yield Arrow RecordBatches."""
        for part in shard.descriptor.parts:
            source = self._source_file(part.source_index, part.path)
            pf = self._get_parquet_file(source)
            if part.end == -1:
                rg_indices = None
            elif part.unit.startswith("rows:"):
                row_group = int(part.unit.split(":", 1)[1])
                offset = 0
                remaining = part.end - part.start
                for batch in pf.iter_batches(
                    row_groups=[row_group],
                    batch_size=self.arrow_batch_size,
                    columns=self.columns_to_read,
                ):
                    batch_rows = int(batch.num_rows)
                    if offset + batch_rows <= part.start:
                        offset += batch_rows
                        continue
                    begin = max(0, part.start - offset)
                    length = min(batch_rows - begin, remaining)
                    if length > 0:
                        yield batch.slice(begin, length)
                        remaining -= length
                    offset += batch_rows
                    if remaining <= 0:
                        break
                continue
            elif self.sharding_mode == "bytes_lazy":
                # Interpret file-part start/end as byte-range planning hints and map to row groups deterministically.
                md = pf.metadata
                if md is None:
                    # If metadata isn't available, we cannot safely map byte-ranges to row groups.
                    # Only allow read-all for the first shard; otherwise we'd duplicate work.
                    if part.start == 0:
                        logger.warning(
                            "Parquet metadata unavailable in bytes_lazy mode for {}; "
                            "falling back to reading full file for first shard only.",
                            part.path,
                        )
                        rg_indices = None
                    else:
                        continue
                else:
                    n = md.num_row_groups
                    if n <= 0:
                        continue

                    file_size = self.fileset.size(part.source_index, part.path)
                    if file_size <= 0:
                        continue

                    a = max(0, part.start)
                    b = min(file_size, part.end)
                    if b <= a:
                        continue

                    cum = 0
                    start_rg: Optional[int] = None
                    end_rg: Optional[int] = None
                    for i in range(n):
                        w = md.row_group(i).total_byte_size
                        w_i = int(w) if w is not None else 0
                        if a <= cum < b:
                            if start_rg is None:
                                start_rg = i
                            end_rg = i + 1
                        cum += w_i

                    if start_rg is None:
                        # If weights are unavailable (all 0/None), we cannot safely map byte-ranges to row groups.
                        # Only allow read-all for the first shard; otherwise we'd duplicate work.
                        if a == 0:
                            logger.warning(
                                "Parquet row-group byte sizes unavailable in bytes_lazy mode for {}; "
                                "falling back to reading full file for first shard only.",
                                part.path,
                            )
                            rg_indices = None
                        else:
                            continue
                    else:
                        assert end_rg is not None
                        rg_indices = list(range(start_rg, end_rg))
            else:
                # start and end are row-group indices
                rg_indices = list(range(part.start, part.end))

            for batch in pf.iter_batches(
                row_groups=rg_indices,
                batch_size=self.arrow_batch_size,
                columns=self.columns_to_read,
            ):
                yield batch


__all__ = ["ParquetReader"]
