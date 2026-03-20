from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc

from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row

ShardDeltaFn = Callable[[dict[str, int]], None]


@dataclass(slots=True)
class ShardDeltaTracker:
    emit_fn: ShardDeltaFn | None
    values: dict[str, int] = field(default_factory=dict)

    def __enter__(self) -> ShardDeltaTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.emit()

    def add(self, shard_id: str, amount: int) -> None:
        if self.emit_fn is None or amount == 0:
            return
        next_value = self.values.get(shard_id, 0) + amount
        if next_value == 0:
            self.values.pop(shard_id, None)
        else:
            self.values[shard_id] = next_value

    def remove_rows(self, rows: Iterable[Row]) -> None:
        if self.emit_fn is None:
            return
        for row in rows:
            if row.shard_id is not None:
                self.add(row.shard_id, -1)

    def emit(self) -> None:
        if self.values and self.emit_fn is not None:
            self.emit_fn(self.values)


def count_rows_by_shard(rows: list[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        shard_id = row.require_shard_id()
        out[shard_id] = out.get(shard_id, 0) + 1
    return out


def count_table_by_shard(table: pa.Table) -> dict[str, int]:
    if SHARD_ID_COLUMN not in table.schema.names:
        return {}
    counts = pc.call_function("value_counts", [table.column(SHARD_ID_COLUMN)])
    values = counts.field("values").to_pylist()
    totals = counts.field("counts").to_pylist()
    return {
        value if isinstance(value, str) else str(value): int(total)
        for value, total in zip(values, totals, strict=True)
        if value is not None
    }


def count_block_by_shard(block: Row | Block) -> dict[str, int]:
    if isinstance(block, Row):
        return {block.require_shard_id(): 1}
    if isinstance(block, Tabular):
        return count_table_by_shard(block.table)
    return count_rows_by_shard(cast(list[Row], block))


def counts_delta(
    *, produced: Mapping[str, int], consumed: Mapping[str, int]
) -> dict[str, int]:
    out: dict[str, int] = {}
    keys = set(consumed) | set(produced)
    for key in keys:
        delta = int(produced.get(key, 0)) - int(consumed.get(key, 0))
        if delta:
            out[key] = delta
    return out


__all__ = [
    "SHARD_ID_COLUMN",
    "ShardDeltaFn",
    "ShardDeltaTracker",
    "count_rows_by_shard",
    "count_table_by_shard",
    "count_block_by_shard",
    "counts_delta",
]
