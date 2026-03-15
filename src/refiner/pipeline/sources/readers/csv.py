from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping
from typing import Any, Optional

import pyarrow.csv as pa_csv
from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
)


class CsvReader(BaseReader):
    """CSV reader sharded by byte ranges (per-file).

    Notes:
        - `list_shards()` only plans byte spans or whole-file atomic shards.
        - `read_shard()` snaps byte-planned shards to newline boundaries before parsing.
        - `multiline_rows=True` disables byte-splitting and keeps each file atomic.
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
        multiline_rows: bool = False,
        encoding: str = "utf-8",
        parse_use_threads: bool = False,
    ):
        """Create a CSV reader.

        Args:
            multiline_rows: If True, keep files atomic and use Python's csv reader
                so embedded newlines inside quoted fields stay correct.
            parse_use_threads: Whether pyarrow's CSV parser may use internal threads
                inside a shard read.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".csv",),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
        )
        self.multiline_rows = multiline_rows
        self.split_by_bytes = not multiline_rows
        self.encoding = encoding
        self.parse_use_threads = parse_use_threads
        self._open_header: Optional[list[str]] = None

    def _get_handle_and_header(self, source_file):
        """Get a cached handle for a resolved file and parse/cache its header row."""
        fh, opened_new = self._get_file_handle(source_file, mode="rb")
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

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        # Multiline CSV needs the Python parser; the normal path stays on Arrow batches.
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        for part in descriptor.parts:
            if self.multiline_rows:
                yield from self._read_shard_python(part)
                continue
            yield from self._read_shard_arrow(part)

    def _bounded_part(
        self, part: FilePart
    ) -> tuple[io.BufferedReader, list[str], int] | None:
        """Open one planned CSV file part after snapping it to whole-record boundaries."""
        source = self.fileset.resolve_file(part.source_index, part.path)
        header = self._get_handle_and_header(source)[1]
        # CSV keeps the same byte-planned shard model as JSONL, but snaps to newline-safe record starts here.
        aligned = self._open_aligned_byte_span(part)
        if aligned is None:
            return None
        _, raw, start = aligned
        return raw, header, start

    def _read_shard_arrow(self, part: FilePart) -> Iterator[SourceUnit]:
        source = self.fileset.resolve_file(part.source_index, part.path)
        if part.end == -1:
            # Whole-file read (e.g. compressed/non-splittable): let Arrow parse
            # directly and stream RecordBatch objects downstream.
            with source.open(
                mode="rb",
                compression="infer",
            ) as raw:
                reader = pa_csv.open_csv(
                    raw,
                    read_options=pa_csv.ReadOptions(
                        use_threads=self.parse_use_threads,
                        encoding=self.encoding,
                    ),
                    parse_options=pa_csv.ParseOptions(
                        newlines_in_values=False,
                    ),
                )
                for batch in reader:
                    yield batch
            return

        bounded = self._bounded_part(part)
        if bounded is None:
            return
        raw, header, start = bounded
        # Non-zero-start shards don't contain headers; reuse cached header names.
        if start == 0:
            read_options = pa_csv.ReadOptions(
                use_threads=self.parse_use_threads,
                encoding=self.encoding,
            )
        else:
            read_options = pa_csv.ReadOptions(
                use_threads=self.parse_use_threads,
                encoding=self.encoding,
                column_names=header,
            )

        reader = pa_csv.open_csv(
            raw,
            read_options=read_options,
            parse_options=pa_csv.ParseOptions(
                newlines_in_values=False,
            ),
        )
        for batch in reader:
            yield batch

    def _read_shard_python(self, part: FilePart) -> Iterator[SourceUnit]:
        source = self.fileset.resolve_file(part.source_index, part.path)
        if part.end == -1:
            with source.open(
                mode="rt",
                compression="infer",
                encoding=self.encoding,
                newline="",
            ) as tf:
                reader = csv.DictReader(tf)
                for row in reader:
                    yield DictRow(row)
            return

        bounded = self._bounded_part(part)
        if bounded is None:
            return
        raw, header, start = bounded
        # Python fallback preserves behavior for multiline quoted records.
        tf = io.TextIOWrapper(raw, encoding=self.encoding, newline="")
        reader = csv.reader(tf)
        if start == 0:
            try:
                next(reader)
            except StopIteration:
                return

        for fields in reader:
            if not fields:
                continue
            if len(fields) < len(header):
                fields = list(fields) + [None] * (len(header) - len(fields))
            if len(fields) > len(header):
                fields = fields[: len(header)]
            yield DictRow(dict(zip(header, fields)))


__all__ = ["CsvReader"]
