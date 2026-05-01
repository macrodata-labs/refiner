from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.json as pa_json
from fsspec import AbstractFileSystem

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import (
    DTypeMapping,
    apply_dtypes_to_table,
)
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
)


class JsonReader(BaseReader):
    """JSON reader for line-delimited JSON or one document per file.

    Notes:
        - With `lines=True`, this reader assumes one JSON value per line
          (newline-delimited JSON) and shards splittable files by byte range.
        - With `lines=False`, each file is one sample and is planned atomically.
    """

    name = "read_json"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: str | None = "file_path",
        parse_use_threads: bool = False,
        dtypes: DTypeMapping | None = None,
        lines: bool = False,
    ):
        """Create a JSON reader.

        Args:
            lines: If True, read newline-delimited JSON. If False, read each
                matched file as a single JSON value.
            parse_use_threads: Whether pyarrow's JSON parser may use internal threads
                inside a shard read.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(
                (".jsonl", ".jsonl.gz", ".ndjson", ".jsonlines")
                if lines
                else (".json", ".json.gz")
            ),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=lines,
            dtypes=dtypes,
        )
        self.lines = lines
        self.parse_use_threads = parse_use_threads

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one planned JSON shard."""
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            if not self.lines:
                with source.open(mode="rb", compression="infer") as raw:
                    value = orjson.loads(raw.read())
                row = value if isinstance(value, dict) else {"value": value}
                row = self._with_file_path(row, source)
                if row:
                    table = pa.Table.from_pylist([row])
                else:
                    table = pa.table({name: [None] for name in self.dtypes or {}})
                yield Tabular(apply_dtypes_to_table(table, self.dtypes, strict=False))
                continue

            if part.end == -1:
                with source.open(
                    mode="rb",
                    compression="infer",
                ) as raw:
                    reader = pa_json.open_json(
                        raw,
                        read_options=pa_json.ReadOptions(
                            use_threads=self.parse_use_threads
                        ),
                    )
                    for batch in reader:
                        yield Tabular(
                            apply_dtypes_to_table(
                                self._table_with_file_path(
                                    pa.Table.from_batches([batch]), source
                                ),
                                self.dtypes,
                                strict=False,
                            )
                        )
                continue

            aligned = self._open_aligned_byte_span(part)
            if aligned is None:
                continue
            _, raw, _ = aligned
            reader = pa_json.open_json(
                raw,
                read_options=pa_json.ReadOptions(use_threads=self.parse_use_threads),
            )
            for batch in reader:
                yield Tabular(
                    apply_dtypes_to_table(
                        self._table_with_file_path(
                            pa.Table.from_batches([batch]), source
                        ),
                        self.dtypes,
                        strict=False,
                    )
                )


__all__ = ["JsonReader"]
