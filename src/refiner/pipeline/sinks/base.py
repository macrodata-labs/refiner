from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from typing import cast

import pyarrow as pa

from refiner.execution.tracking.shards import SHARD_ID_COLUMN, count_block_by_shard
from refiner.io.datafolder import DataFolder
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.row import Row

Block = list[Row] | Tabular
ShardedBlock = list[Row] | Tabular
ShardCounts = dict[str, int]


class BaseSink(ABC):
    @abstractmethod
    def write_block(self, block: Block) -> ShardCounts:
        raise NotImplementedError

    def describe(self) -> tuple[str, str, dict[str, Any] | None] | None:
        return None

    def on_shard_complete(self, shard_id: str) -> None:
        del shard_id

    def close(self) -> None:
        return None


class NullSink(BaseSink):
    def write_block(self, block: Block) -> ShardCounts:
        return count_block_by_shard(block)


def describe_datafolder_path(value: Any) -> str:
    folder = DataFolder.resolve(value)
    return str(folder.fs.unstrip_protocol(folder.path))


def split_block_by_shard(block: Block) -> tuple[dict[str, ShardedBlock], ShardCounts]:
    if not isinstance(block, Tabular):
        rows_by_shard: dict[str, list[Row]] = {}
        counts: ShardCounts = {}
        row_block = cast(list[Row], block)
        for row in row_block:
            shard_id = row.require_shard_id()
            rows_by_shard.setdefault(shard_id, []).append(row)
            counts[shard_id] = counts.get(shard_id, 0) + 1
        return dict(rows_by_shard), counts

    table = block.table
    if SHARD_ID_COLUMN not in table.schema.names:
        raise ValueError("tabular sink input is missing __shard_id")

    shard_indices: dict[str, list[int]] = {}
    for idx, shard_id in enumerate(table.column(SHARD_ID_COLUMN).to_pylist()):
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError("tabular sink input has invalid __shard_id")
        shard_indices.setdefault(shard_id, []).append(idx)

    data_table = table.drop_columns([SHARD_ID_COLUMN])
    tables_by_shard: dict[str, Tabular] = {}
    counts: ShardCounts = {}
    for shard_id, indices in shard_indices.items():
        counts[shard_id] = len(indices)
        if len(indices) == data_table.num_rows:
            tables_by_shard[shard_id] = block.with_table(data_table)
        else:
            tables_by_shard[shard_id] = block.with_table(
                data_table.take(pa.array(indices, type=pa.int64()))
            )
    return dict(tables_by_shard), counts


__all__ = [
    "BaseSink",
    "NullSink",
    "Block",
    "ShardCounts",
    "describe_datafolder_path",
    "split_block_by_shard",
]
