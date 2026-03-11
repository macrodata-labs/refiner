from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import IO

import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike

from .base import (
    BaseSink,
    Block,
    ShardCounts,
    split_block_by_shard,
)


class JsonlSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}.jsonl",
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self._files: dict[str, IO[str]] = {}
        self._encoder = json.JSONEncoder(
            ensure_ascii=True,
            separators=(",", ":"),
        )

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(shard_id=shard_id)

    def _file(self, shard_id: str) -> IO[str]:
        file = self._files.get(shard_id)
        if file is not None:
            return file
        file = self.output.open(self._relpath(shard_id), mode="at", encoding="utf-8")
        self._files[shard_id] = file
        return file

    def _write_rows(self, shard_id: str, rows: Iterable[Mapping[str, object]]) -> None:
        file = self._file(shard_id)
        for row in rows:
            file.write(self._encoder.encode(row))
            file.write("\n")

    def _write_table_rows(self, shard_id: str, table: pa.Table) -> None:
        for batch in table.to_batches(max_chunksize=4096):
            self._write_rows(shard_id, batch.to_pylist())

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)
        for shard_id, shard_block in blocks_by_shard.items():
            if isinstance(shard_block, pa.Table):
                self._write_table_rows(shard_id, shard_block)
                continue
            self._write_rows(shard_id, (row.to_dict() for row in shard_block))
        return counts

    def on_shard_complete(self, shard_id: str) -> None:
        file = self._files.pop(shard_id, None)
        if file is not None:
            file.close()

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()


__all__ = ["JsonlSink"]
