from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Optional

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    clamp_target_bytes,
)

_LOG = logging.getLogger(__name__)


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

        self._open_pf: Optional[pq.ParquetFile] = None

    def _get_parquet_file(self, path: str) -> pq.ParquetFile:
        """Get or open a cached ParquetFile for the current path (single-open-file policy)."""
        fh, opened_new = self._get_file_handle(path, mode="rb")
        if opened_new or self._open_pf is None:
            # fsspec file objects are usually seekable and acceptable for pyarrow.parquet
            self._open_pf = pq.ParquetFile(fh)
        return self._open_pf

    def list_shards(self) -> list[Shard]:
        """List shards for all resolved input files.

        For Parquet:
            - If `sharding_mode == "rowgroups"`:
                - `Shard.start` / `Shard.end` are **row-group indices**.
                - Row-group ranges are half-open: `[start, end)`.
            - If `sharding_mode == "bytes_lazy"`:
                - `Shard.start` / `Shard.end` are **byte offsets** for planning only.
                - `read_shard()` will load metadata and map the byte-range deterministically onto row groups.
            - `end == -1` is used as a sentinel meaning \"read the whole file\".
        """
        shards: list[Shard] = []
        for path in self.files:
            size = self.fileset.size(path)

            # Fast path: if the whole file is smaller than the target, keep it as a single shard.
            if size <= self.target_shard_bytes:
                shards.append(Shard(path=path, start=0, end=-1))
                continue

            if self.sharding_mode == "bytes_lazy":
                # Plan byte-range shards without reading parquet metadata; map to row groups at runtime.
                start = 0
                while start < size:
                    end = min(size, start + self.target_shard_bytes)
                    shards.append(Shard(path=path, start=start, end=end))
                    start = end
                continue

            pf = self._get_parquet_file(path)
            md = pf.metadata
            if md is None:
                # no metadata? treat as 1 shard (best-effort)
                shards.append(Shard(path=path, start=0, end=-1))
                continue

            n = md.num_row_groups
            if n <= 0:
                # empty file? no shards
                continue

            start = 0
            acc = 0
            for rg in range(n):
                # this doesn't perform extra I/O btw as its in the metadata
                rg_bytes = md.row_group(rg).total_byte_size
                rg_bytes = int(rg_bytes) if rg_bytes is not None else 0

                # always include at least one row group per shard
                if rg == start:
                    acc = rg_bytes
                    continue

                if acc >= self.target_shard_bytes:
                    end = rg
                    shards.append(
                        Shard(
                            path=path,
                            start=start,
                            end=end,
                        )
                    )
                    start = rg
                    acc = rg_bytes
                else:
                    acc += rg_bytes

            # tail shard
            shards.append(
                Shard(
                    path=path,
                    start=start,
                    end=n,
                )
            )

        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read a parquet shard and yield Arrow RecordBatches."""
        pf = self._get_parquet_file(shard.path)

        # Determine row groups to read.
        if shard.end == -1:
            rg_indices = None
        elif self.sharding_mode == "bytes_lazy":
            # Interpret shard.start/end as byte-range planning hints and map to row groups deterministically.
            md = pf.metadata
            if md is None:
                # If metadata isn't available, we cannot safely map byte-ranges to row groups.
                # Only allow read-all for the first shard; otherwise we'd duplicate work.
                if shard.start == 0:
                    _LOG.warning(
                        "Parquet metadata unavailable in bytes_lazy mode for %s; "
                        "falling back to reading full file for first shard only.",
                        shard.path,
                    )
                    rg_indices = None
                else:
                    return
            else:
                n = md.num_row_groups
                if n <= 0:
                    return

                file_size = self.fileset.size(shard.path)
                if file_size <= 0:
                    return

                a = max(0, shard.start)
                b = min(file_size, shard.end)
                if b <= a:
                    return

                # Map the planned byte range [a, b) to a contiguous row-group range using row-group byte sizes
                # as weights. This is approximate (row-group bytes won't sum exactly to file_size), but it's
                # deterministic and avoids planning-time metadata reads.
                cum = 0
                start_rg: Optional[int] = None
                end_rg: Optional[int] = None
                for i in range(n):
                    w = md.row_group(i).total_byte_size
                    w_i = int(w) if w is not None else 0
                    # [!] Include row groups whose *start* offset is within [a, b).
                    if a <= cum < b:
                        if start_rg is None:
                            start_rg = i
                        end_rg = i + 1

                    cum += w_i

                if start_rg is None:
                    # If weights are unavailable (all 0/None), we cannot safely map byte-ranges to row groups.
                    # Only allow read-all for the first shard; otherwise we'd duplicate work.
                    if a == 0:
                        _LOG.warning(
                            "Parquet row-group byte sizes unavailable in bytes_lazy mode for %s; "
                            "falling back to reading full file for first shard only.",
                            shard.path,
                        )
                        rg_indices = None
                    else:
                        return
                else:
                    assert end_rg is not None
                    rg_indices = list(range(start_rg, end_rg))
        else:
            # start and end are row-group indices
            rg_indices = list(range(shard.start, shard.end))

        # Stream record batches so we don't load the whole shard into memory at once.
        for batch in pf.iter_batches(
            row_groups=rg_indices,
            batch_size=self.arrow_batch_size,
            columns=self.columns_to_read,
        ):
            yield batch


__all__ = ["ParquetReader"]
