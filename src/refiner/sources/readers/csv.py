from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Optional

from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike

from .base import BaseReader, Shard
from ..row import DictRow, Row
from .utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    BoundedBinaryReader,
    align_byte_range_to_newlines,
    clamp_target_bytes,
    is_splittable_by_bytes,
)


class CsvReader(BaseReader):
    """CSV reader sharded by byte ranges (per-file)."""

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        multiline_rows: bool = False,
        encoding: str = "utf-8",
        sharding_mode: Literal["bytes_lazy", "scan"] = "bytes_lazy",
    ):
        """Create a CSV reader.

        Notes:
            - Shards are byte ranges within each file.
            - `sharding_mode="bytes_lazy"` (default) plans shards by bytes without scanning; `read_shard()` aligns
              planned ranges to newline boundaries at runtime. Only valid when `multiline_rows=False`.
            - `sharding_mode="scan"` performs a full scan in `list_shards()` to find record boundaries.
            - `multiline_rows=False` assumes one record per newline.
            - `multiline_rows=True` supports embedded newlines inside quotes and forces `sharding_mode="scan"`.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".csv",),
        )
        self.target_shard_bytes = clamp_target_bytes(target_shard_bytes)
        self.multiline_rows = multiline_rows
        self.encoding = encoding
        self._open_header: Optional[list[str]] = None
        if sharding_mode not in ("bytes_lazy", "scan"):
            raise ValueError("sharding_mode must be 'bytes_lazy' or 'scan'")
        self.sharding_mode = sharding_mode

    def _get_handle_and_header(self, path: str):
        """Get a cached handle for `path` and parse/cache its header row."""
        fh, opened_new = self._get_file_handle(path, mode="rb")
        if opened_new:
            self._open_header = None
        if self._open_header is not None:
            return fh, self._open_header

        # Parse header from the beginning of the file.
        try:
            if hasattr(fh, "seek"):
                fh.seek(0)
        except Exception:
            pass

        # Use fsspec compression inference for correctness for compressed files.
        # For splittable (uncompressed) files, this is still fine.
        tf = io.TextIOWrapper(fh, encoding=self.encoding, newline="")
        reader = csv.reader(tf)
        header = next(reader)
        # Prevent the TextIOWrapper from closing the shared underlying file handle.
        try:
            tf.detach()
        except Exception:
            pass
        self._open_header = list(header)
        return fh, self._open_header

    def list_shards(self) -> list[Shard]:
        """List shards for all resolved input files.

        For CSV:
            - `Shard.start` / `Shard.end` are **byte offsets** within the file.
            - Ranges are half-open: `[start, end)`.
            - `end == -1` is used as a sentinel meaning \"read the whole file\" (non-splittable inputs).
        """
        shards: list[Shard] = []
        for path in self.files:
            if not is_splittable_by_bytes(self.fs, path):
                # one shard per file
                shards.append(Shard(path=path, start=0, end=-1))
                continue

            size = self.fileset.size(path)
            if size <= self.target_shard_bytes:
                shards.append(Shard(path=path, start=0, end=size))
                continue

            mode = self.sharding_mode
            if self.multiline_rows and mode == "bytes_lazy":
                mode = "scan"

            if mode == "bytes_lazy":
                start = 0
                while start < size:
                    end = min(size, start + self.target_shard_bytes)
                    shards.append(Shard(path=path, start=start, end=end))
                    start = end
            else:
                # Streaming scan to find cut points.
                with self.fs.open(path, mode="rb") as f:
                    shard_start = 0
                    pos = 0
                    next_cut_at = self.target_shard_bytes

                    in_quotes = False
                    buf = f.read(1024 * 1024)
                    while buf:
                        i = 0
                        n = len(buf)
                        while i < n:
                            b = buf[i]
                            if self.multiline_rows:
                                if b == 34:  # ord('"')
                                    if in_quotes and i + 1 < n and buf[i + 1] == 34:
                                        i += 2
                                        pos += 2
                                        continue
                                    in_quotes = not in_quotes
                            if b == 10 and (
                                not self.multiline_rows or not in_quotes
                            ):  # '\n'
                                # newline ends a record in non-multiline mode, or ends a record when not in quotes.
                                if pos + 1 >= next_cut_at:
                                    cut = pos + 1  # start next shard after newline
                                    shards.append(
                                        Shard(
                                            path=path,
                                            start=shard_start,
                                            end=cut,
                                        )
                                    )
                                    shard_start = cut
                                    next_cut_at = shard_start + self.target_shard_bytes
                            i += 1
                            pos += 1
                        buf = f.read(1024 * 1024)

                    if shard_start < size:
                        shards.append(
                            Shard(
                                path=path,
                                start=shard_start,
                                end=size,
                            )
                        )

        return shards

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        """Read a CSV shard and yield `Row` objects.

        Notes:
            - Yields `DictRow` rows (dict-backed).
            - For `end == -1` (non-splittable inputs), the entire file is read with `compression='infer'`.
            - For byte-range shards, shard boundaries are assumed to be at record boundaries
              (enforced by `list_shards()` when sharding is enabled).
        """
        # Non-splittable inputs: read the entire file with decompression if needed.
        if shard.end == -1:
            with self.fs.open(
                shard.path,
                mode="rt",
                compression="infer",
                encoding=self.encoding,
                newline="",
            ) as tf:
                reader = csv.DictReader(tf)
                for row in reader:
                    yield DictRow(row)
            return

        # Splittable: reuse single open binary file if possible, seek to shard.start.
        fh, header = self._get_handle_and_header(shard.path)
        size = self.fileset.size(shard.path)

        mode = self.sharding_mode
        if self.multiline_rows and mode == "bytes_lazy":
            # TODO: warning
            mode = "scan"

        start = shard.start
        end = shard.end

        if mode == "bytes_lazy":
            # Align planned [start, end) to newline boundaries so we include lines whose *start* is in [start, end).
            # We interpret record boundaries as '\n' (valid only when multiline_rows=False).
            aligned = align_byte_range_to_newlines(fh, start=start, end=end, size=size)
            if aligned is None:
                return
            start, end = aligned

        # Seek to shard start for this shard.
        try:
            fh.seek(start)
        except Exception:
            # fallback: reopen (still one-at-a-time)
            self._open_header = None
            fh, _ = self._get_file_handle(shard.path, mode="rb", force_reopen=True)
            fh.seek(start)

        # Build a bounded text stream for this shard. We assume shard boundaries are at record boundaries.
        raw = io.BufferedReader(BoundedBinaryReader(fh, end - start))
        tf = io.TextIOWrapper(raw, encoding=self.encoding, newline="")
        reader = csv.reader(tf)

        # In the first shard, the first record is the header; otherwise treat all records as data.
        if start == 0:
            try:
                next(reader)  # consume header already cached
            except StopIteration:
                return

        for fields in reader:
            if not fields:
                continue
            # If row has fewer fields than header, pad with None; if more, truncate.
            if len(fields) < len(header):
                fields = list(fields) + [None] * (len(header) - len(fields))
            if len(fields) > len(header):
                fields = fields[: len(header)]

            yield DictRow(dict(zip(header, fields)))


__all__ = ["CsvReader"]
