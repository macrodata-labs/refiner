from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pyarrow as pa

from refiner.ledger.shard_tracking import SHARD_ID_COLUMN, count_block_by_shard
from refiner.runtime.types import TabularBlock
from refiner.sources.row import Row

Block = list[Row] | TabularBlock
ShardedBlock = list[Row] | pa.Table
ShardCounts = dict[str, int]


@runtime_checkable
class ShardCompletionListener(Protocol):
    def on_shard_complete(self, shard_id: str) -> None: ...


class BaseSink(ABC):
    @abstractmethod
    def write_block(self, block: Block) -> ShardCounts:
        raise NotImplementedError

    def on_shard_complete(self, shard_id: str) -> None:
        del shard_id

    def close(self) -> None:
        return None


class NullSink(BaseSink):
    def write_block(self, block: Block) -> ShardCounts:
        return count_block_by_shard(block)


def split_block_by_shard(block: Block) -> tuple[dict[str, ShardedBlock], ShardCounts]:
    if isinstance(block, list):
        rows_by_shard: dict[str, list[Row]] = {}
        counts: ShardCounts = {}
        for row in block:
            shard_id = row.require_shard_id()
            rows_by_shard.setdefault(shard_id, []).append(row)
            counts[shard_id] = counts.get(shard_id, 0) + 1
        return rows_by_shard, counts

    table = block if isinstance(block, pa.Table) else pa.Table.from_batches([block])
    if SHARD_ID_COLUMN not in table.schema.names:
        raise ValueError("tabular sink input is missing __shard_id")

    shard_indices: dict[str, list[int]] = {}
    for idx, shard_id in enumerate(table.column(SHARD_ID_COLUMN).to_pylist()):
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError("tabular sink input has invalid __shard_id")
        shard_indices.setdefault(shard_id, []).append(idx)

    data_table = table.drop_columns([SHARD_ID_COLUMN])
    tables_by_shard: dict[str, pa.Table] = {}
    counts: ShardCounts = {}
    for shard_id, indices in shard_indices.items():
        counts[shard_id] = len(indices)
        if len(indices) == data_table.num_rows:
            tables_by_shard[shard_id] = data_table
            continue
        tables_by_shard[shard_id] = data_table.take(pa.array(indices, type=pa.int64()))
    return tables_by_shard, counts


def negate_counts(counts: ShardCounts) -> ShardCounts:
    return {shard_id: -count for shard_id, count in counts.items() if count}


__all__ = [
    "Block",
    "ShardCounts",
    "ShardCompletionListener",
    "BaseSink",
    "NullSink",
    "split_block_by_shard",
    "negate_counts",
]
