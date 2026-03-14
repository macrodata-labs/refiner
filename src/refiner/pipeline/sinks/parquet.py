from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from refiner.execution.operators.vectorized import rows_to_table
from refiner.execution.tracking.shards import SHARD_ID_COLUMN
from refiner.io.datafolder import DataFolder, DataFolderLike

from refiner.pipeline.sinks.base import (
    BaseSink,
    Block,
    ShardCounts,
    split_block_by_shard,
)


class ParquetSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}.parquet",
        compression: str | None = None,
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.compression = compression
        self._writers: dict[str, pq.ParquetWriter] = {}

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(shard_id=shard_id)

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
                shard_block
                if isinstance(shard_block, pa.Table)
                else rows_to_table(shard_block)
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


__all__ = ["ParquetSink"]
