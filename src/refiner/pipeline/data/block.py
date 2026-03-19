from __future__ import annotations

from typing import cast

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular

Block = list[Row] | Tabular
StreamItem = Row | Block


def split_block_by_shard(block: Block) -> tuple[dict[str, Block], dict[str, int]]:
    if not isinstance(block, Tabular):
        rows_by_shard: dict[str, list[Row]] = {}
        row_block = cast(list[Row], block)
        for row in row_block:
            shard_id = row.require_shard_id()
            rows_by_shard.setdefault(shard_id, []).append(row)
        return dict(rows_by_shard), {
            shard_id: len(rows) for shard_id, rows in rows_by_shard.items()
        }

    if block.shard_idx is None:
        raise ValueError("tabular sink input is missing __shard_id")
    shard_column = block.columns[block.shard_idx].combine_chunks()
    encoded = shard_column.dictionary_encode()
    dictionary = [cast(str, shard_id) for shard_id in encoded.dictionary.to_pylist()]
    if any(not isinstance(shard_id, str) or not shard_id for shard_id in dictionary):
        raise ValueError("tabular sink input has invalid __shard_id")
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    data_table = block.table.drop_columns([SHARD_ID_COLUMN])
    codes_monotonic = bool(np.all(codes[1:] >= codes[:-1]))
    return (
        _split_tabular_by_shard_sorted(block, data_table, codes, dictionary)
        if _use_sorted_code_path(
            unique_shards=len(dictionary),
            num_rows=data_table.num_rows,
            num_columns=data_table.num_columns,
            codes_monotonic=codes_monotonic,
        )
        else _split_tabular_by_shard_scanned(block, data_table, codes, dictionary)
    )


def _use_sorted_code_path(
    *,
    unique_shards: int,
    num_rows: int,
    num_columns: int,
    codes_monotonic: bool,
) -> bool:
    if unique_shards <= 1:
        return False
    rows_per_shard = num_rows / unique_shards
    if codes_monotonic:
        # Grouped shard layouts benefit from the sorted path earlier, especially as
        # payload width grows and repeated take() calls get more expensive.
        rows_per_shard_threshold = min(192.0, 128.0 + (2.0 * num_columns))
        return rows_per_shard <= rows_per_shard_threshold
    # Mixed shard layouts stay on the simpler scanned path until shard cardinality
    # is already high; width-aware switching regressed the interleaved cases.
    # this magic number was the result of many benchmarks run by codex, seems to work ok
    return unique_shards >= 768


def _split_tabular_by_shard_scanned(
    block: Tabular,
    data_table: pa.Table,
    codes: np.ndarray,
    dictionary: list[str],
) -> tuple[dict[str, Block], dict[str, int]]:
    tables_by_shard: dict[str, Tabular] = {}
    counts: dict[str, int] = {}
    for code, shard_id in enumerate(dictionary):
        indices = np.nonzero(codes == code)[0]
        if indices.size == 0:
            continue
        counts[shard_id] = int(indices.size)
        if indices.size == data_table.num_rows:
            tables_by_shard[shard_id] = block.with_table(data_table)
        else:
            tables_by_shard[shard_id] = block.with_table(
                data_table.take(pa.array(indices, type=pa.int64()))
            )
    return dict(tables_by_shard), counts


def _split_tabular_by_shard_sorted(
    block: Tabular,
    data_table: pa.Table,
    codes: np.ndarray,
    dictionary: list[str],
) -> tuple[dict[str, Block], dict[str, int]]:
    order = (
        np.arange(codes.shape[0], dtype=np.int64)
        if np.all(codes[1:] >= codes[:-1])
        else np.argsort(codes, kind="stable")
    )
    ordered_codes = codes[order]
    tables_by_shard: dict[str, Tabular] = {}
    counts: dict[str, int] = {}
    start = 0
    while start < order.size:
        code = int(ordered_codes[start])
        end = start + 1
        while end < order.size and int(ordered_codes[end]) == code:
            end += 1
        shard_id = dictionary[code]
        shard_indices = order[start:end]
        counts[shard_id] = int(shard_indices.size)
        if shard_indices.size == data_table.num_rows:
            tables_by_shard[shard_id] = block.with_table(data_table)
        elif np.all(shard_indices[1:] == shard_indices[:-1] + 1):
            tables_by_shard[shard_id] = block.with_table(
                data_table.slice(int(shard_indices[0]), int(shard_indices.size))
            )
        else:
            tables_by_shard[shard_id] = block.with_table(
                data_table.take(pa.array(shard_indices, type=pa.int64()))
            )
        start = end
    return dict(tables_by_shard), counts


__all__ = ["Block", "StreamItem", "split_block_by_shard"]
