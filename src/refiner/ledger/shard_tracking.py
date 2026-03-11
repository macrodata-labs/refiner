from __future__ import annotations

from collections.abc import Iterable, Mapping

import pyarrow as pa

from refiner.sources.row import Row

SHARD_ID_COLUMN = "__shard_id"


def count_rows_by_shard(rows: Iterable[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        shard = row.require_shard_id()
        out[shard] = out.get(shard, 0) + 1
    return out


def count_tabular_by_shard(block: pa.RecordBatch | pa.Table) -> dict[str, int]:
    if SHARD_ID_COLUMN not in block.schema.names:
        return {}
    out: dict[str, int] = {}
    for key in block.column(SHARD_ID_COLUMN).to_pylist():
        if key is None:
            continue
        shard = key if isinstance(key, str) else str(key)
        out[shard] = out.get(shard, 0) + 1
    return out


def count_block_by_shard(
    block: Row | list[Row] | pa.RecordBatch | pa.Table,
) -> dict[str, int]:
    if isinstance(block, Row):
        return {block.require_shard_id(): 1}
    if isinstance(block, list):
        return count_rows_by_shard(block)
    if isinstance(block, (pa.RecordBatch, pa.Table)):
        return count_tabular_by_shard(block)
    raise TypeError(f"unsupported block type: {type(block)!r}")


def counts_delta(
    *, produced: Mapping[str, int], consumed: Mapping[str, int]
) -> dict[str, int]:
    out: dict[str, int] = {}
    keys = set(consumed.keys()) | set(produced.keys())
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
