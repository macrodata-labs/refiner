from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row

SHARD_ID_COLUMN = "__shard_id"


def count_rows_by_shard(rows: list[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        shard_id = row.require_shard_id()
        out[shard_id] = out.get(shard_id, 0) + 1
    return out


def count_tabular_by_shard(block: Tabular) -> dict[str, int]:
    table = block.table
    if SHARD_ID_COLUMN not in table.schema.names:
        return {}
    out: dict[str, int] = {}
    for key in table.column(SHARD_ID_COLUMN).to_pylist():
        if key is None:
            continue
        shard_id = key if isinstance(key, str) else str(key)
        out[shard_id] = out.get(shard_id, 0) + 1
    return out


def count_block_by_shard(block: Row | Block) -> dict[str, int]:
    if isinstance(block, Row):
        return {block.require_shard_id(): 1}
    if isinstance(block, Tabular):
        return count_tabular_by_shard(block)
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
    "count_rows_by_shard",
    "count_tabular_by_shard",
    "count_block_by_shard",
    "counts_delta",
]
