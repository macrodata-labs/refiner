from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from typing import Any

from fsspec import AbstractFileSystem
import pyarrow.json as pa_json

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    BoundedBinaryReader,
    align_byte_range_to_newlines,
    clamp_target_bytes,
    is_splittable_by_bytes,
)


class JsonlReader(BaseReader):
    """JSONL / NDJSON reader sharded by byte ranges (per-file).

    Notes:
        - This reader assumes one JSON value per line (newline-delimited JSON).
        - Sharding mode is bytes-lazy only: `list_shards()` is cheap, and `read_shard()` aligns to newline boundaries.
    """

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    ):
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".jsonl", ".jsonl.gz", ".ndjson", ".jsonlines"),
        )
        self.target_shard_bytes = clamp_target_bytes(target_shard_bytes)

    def list_shards(self) -> list[Shard]:
        shards: list[Shard] = []
        global_ordinal = 0
        for source_index, files in enumerate(self.fileset.expand_sources()):
            for file in files:
                path = file.abs_path()
                if not is_splittable_by_bytes(file.fs, file.path):
                    shards.append(
                        Shard(
                            path=path,
                            start=0,
                            end=-1,
                            source_index=source_index,
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1
                    continue

                size = self.fileset.size(source_index, path)
                if size <= self.target_shard_bytes:
                    shards.append(
                        Shard(
                            path=path,
                            start=0,
                            end=size,
                            source_index=source_index,
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1
                    continue

                start = 0
                while start < size:
                    end = min(size, start + self.target_shard_bytes)
                    shards.append(
                        Shard(
                            path=path,
                            start=start,
                            end=end,
                            source_index=source_index,
                            global_ordinal=global_ordinal,
                        )
                    )
                    global_ordinal += 1
                    start = end

        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        for part in shard.parts:
            source = self._source_file(part.source_index, part.path)
            if part.end == -1:
                with source.open(
                    mode="rb",
                    compression="infer",
                ) as raw:
                    reader = pa_json.open_json(
                        raw,
                        read_options=pa_json.ReadOptions(use_threads=False),
                    )
                    for batch in reader:
                        yield batch
                continue

            fh, _ = self._get_file_handle(source, mode="rb")
            size = self.fileset.size(part.source_index, part.path)
            aligned = align_byte_range_to_newlines(
                fh, start=part.start, end=part.end, size=size
            )
            if aligned is None:
                continue
            start, end = aligned
            try:
                fh.seek(start)
            except Exception:
                fh, _ = self._get_file_handle(source, mode="rb", force_reopen=True)
                fh.seek(start)

            raw = io.BufferedReader(BoundedBinaryReader(fh, end - start))
            reader = pa_json.open_json(
                raw,
                read_options=pa_json.ReadOptions(use_threads=False),
            )
            for batch in reader:
                yield batch


__all__ = ["JsonlReader"]
