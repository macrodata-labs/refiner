from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.tracking.shards import SHARD_ID_COLUMN
from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import TabularBlock

from refiner.pipeline.sinks.base import (
    BaseSink,
    Block,
    ShardCounts,
    describe_datafolder_path,
    split_block_by_shard,
)
from refiner.worker.context import get_active_run_handle


class ParquetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.parquet",
        compression: str | None = None,
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.compression = compression
        self._writers: dict[str, pq.ParquetWriter] = {}

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_run_handle().worker_token,
        )

    def _writer(self, shard_id: str, schema: pa.Schema) -> pq.ParquetWriter:
        writer = self._writers.get(shard_id)
        if writer is not None:
            return writer
        sink = self.output.open(self._relpath(shard_id), mode="wb")
        writer = pq.ParquetWriter(sink, schema, compression=self.compression)
        self._writers[shard_id] = writer
        return writer

    def write_block(self, block: Block) -> ShardCounts:
        blocks_by_shard, counts = split_block_by_shard(block)
        for shard_id, shard_block in blocks_by_shard.items():
            table = (
                shard_block.table
                if isinstance(shard_block, TabularBlock)
                else TabularBlock.from_rows(shard_block).table
            )
            if SHARD_ID_COLUMN in table.schema.names:
                table = table.drop_columns([SHARD_ID_COLUMN])
            self._writer(shard_id, table.schema).write_table(table)
        return counts

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": describe_datafolder_path(self.output),
            "filename_template": self.filename_template,
        }
        if self.compression is not None:
            args["compression"] = self.compression
        return ("write_parquet", "writer", args)


__all__ = ["ParquetSink"]
