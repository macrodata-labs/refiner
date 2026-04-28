from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from fsspec import AbstractFileSystem
import pyarrow as pa
import pyarrow.json as pa_json

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.datatype import (
    DTypeMapping,
    apply_dtypes_to_table,
    schema_with_dtypes,
)
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
)


class JsonlReader(BaseReader):
    """JSONL / NDJSON reader sharded by byte ranges (per-file).

    Notes:
        - This reader assumes one JSON value per line (newline-delimited JSON).
        - `list_shards()` only plans byte spans; `read_shard()` turns them into whole-line reads.
        - Atomic files (`end=-1`) are read whole and never byte-sliced.
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
        file_path_column: str | None = "file_path",
        parse_use_threads: bool = False,
        dtypes: DTypeMapping | None = None,
    ):
        """Create a JSONL reader.

        Args:
            parse_use_threads: Whether pyarrow's JSON parser may use internal threads
                inside a shard read.
        """
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".jsonl", ".jsonl.gz", ".ndjson", ".jsonlines"),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
        )
        self.parse_use_threads = parse_use_threads
        self.dtypes = dtypes

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read one planned JSONL shard by snapping file parts to newline boundaries."""
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
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

    @property
    def schema(self) -> pa.Schema | None:
        return schema_with_dtypes(None, self.dtypes)


__all__ = ["JsonlReader"]
