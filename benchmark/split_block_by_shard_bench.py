from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyarrow as pa

from refiner.execution.tracking.shards import SHARD_ID_COLUMN
from refiner.pipeline.data.block import Block, split_block_by_shard
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.tabular import Tabular


@dataclass(frozen=True)
class BenchCase:
    name: str
    block: list[Row] | Tabular


def _bench(
    fn: Callable[[list[Row] | Tabular], object],
    block: list[Row] | Tabular,
    *,
    iterations: int,
) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(block)
        samples.append((time.perf_counter() - start) * 1_000)
    return statistics.mean(samples), statistics.stdev(samples)


def _row_block(num_rows: int, num_shards: int) -> list[Row]:
    rows: list[Row] = []
    for row_idx in range(num_rows):
        shard_id = f"s{row_idx % num_shards:03d}"
        rows.append(DictRow({"value": row_idx}, shard_id=shard_id))
    return rows


def _tabular_block(num_rows: int, num_shards: int, *, grouped: bool) -> Tabular:
    shard_ids = [f"s{idx:03d}" for idx in range(num_shards)]
    if grouped:
        per_shard = max(1, num_rows // num_shards)
        shards = [
            shard_ids[min(num_shards - 1, row_idx // per_shard)]
            for row_idx in range(num_rows)
        ]
    else:
        shards = [shard_ids[row_idx % num_shards] for row_idx in range(num_rows)]
    return Tabular(
        pa.table(
            {
                SHARD_ID_COLUMN: shards,
                "value": list(range(num_rows)),
                "value2": [row_idx * 2 for row_idx in range(num_rows)],
            }
        )
    )


def _cases() -> list[BenchCase]:
    return [
        BenchCase("rows/10k/10shards", _row_block(10_000, 10)),
        BenchCase(
            "tabular-grouped/10k/10shards", _tabular_block(10_000, 10, grouped=True)
        ),
        BenchCase(
            "tabular-mixed/10k/10shards", _tabular_block(10_000, 10, grouped=False)
        ),
        BenchCase("rows/100k/100shards", _row_block(100_000, 100)),
        BenchCase(
            "tabular-grouped/100k/100shards", _tabular_block(100_000, 100, grouped=True)
        ),
        BenchCase(
            "tabular-mixed/100k/100shards", _tabular_block(100_000, 100, grouped=False)
        ),
    ]


def _variant_row_len_counts(block: Block) -> object:
    if isinstance(block, Tabular):
        return split_block_by_shard(block)
    rows_by_shard: dict[str, list[Row]] = {}
    for row in block:
        rows_by_shard.setdefault(row.require_shard_id(), []).append(row)
    return rows_by_shard, {
        shard_id: len(rows) for shard_id, rows in rows_by_shard.items()
    }


def _variant_tabular_dict(block: Block) -> object:
    if not isinstance(block, Tabular):
        return split_block_by_shard(block)
    if block.shard_idx is None:
        raise ValueError("tabular sink input is missing __shard_id")
    encoded = block.columns[block.shard_idx].combine_chunks().dictionary_encode()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    dictionary = encoded.dictionary.to_pylist()
    if any(not isinstance(item, str) or not item for item in dictionary):
        raise ValueError("tabular sink input has invalid __shard_id")
    data_table = block.table.drop_columns([SHARD_ID_COLUMN])
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
    return tables_by_shard, counts


def _variant_tabular_codesort(block: Block) -> object:
    if not isinstance(block, Tabular):
        return split_block_by_shard(block)
    if block.shard_idx is None:
        raise ValueError("tabular sink input is missing __shard_id")
    encoded = block.columns[block.shard_idx].combine_chunks().dictionary_encode()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    dictionary = encoded.dictionary.to_pylist()
    if any(not isinstance(item, str) or not item for item in dictionary):
        raise ValueError("tabular sink input has invalid __shard_id")
    data_table = block.table.drop_columns([SHARD_ID_COLUMN])
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
    return tables_by_shard, counts


def _variant_tabular_runs(block: Block) -> object:
    if not isinstance(block, Tabular):
        return split_block_by_shard(block)
    if block.shard_idx is None:
        raise ValueError("tabular sink input is missing __shard_id")
    shard_ids = block.columns[block.shard_idx].to_pylist()
    if any(not isinstance(item, str) or not item for item in shard_ids):
        raise ValueError("tabular sink input has invalid __shard_id")
    data_table = block.table.drop_columns([SHARD_ID_COLUMN])
    tables_by_shard: dict[str, Tabular] = {}
    counts: dict[str, int] = {}
    start = 0
    while start < len(shard_ids):
        shard_id = shard_ids[start]
        end = start + 1
        while end < len(shard_ids) and shard_ids[end] == shard_id:
            end += 1
        counts[shard_id] = counts.get(shard_id, 0) + (end - start)
        run = data_table.slice(start, end - start)
        current = tables_by_shard.get(shard_id)
        tables_by_shard[shard_id] = (
            block.with_table(run)
            if current is None
            else current.with_table(pa.concat_tables([current.table, run]))
        )
        start = end
    return tables_by_shard, counts


def main() -> None:
    variants: list[tuple[str, Callable[[Block], object]]] = [
        ("current", split_block_by_shard),
        ("row_len", _variant_row_len_counts),
        ("tab_dict", _variant_tabular_dict),
        ("tab_sort", _variant_tabular_codesort),
        ("tab_runs", _variant_tabular_runs),
    ]
    for case in _cases():
        iterations = 20 if "10k" in case.name else 4
        print(case.name, flush=True)
        for name, fn in variants:
            mean_ms, stdev_ms = _bench(fn, case.block, iterations=iterations)
            print(
                f"  {name:<10} mean={mean_ms:8.3f}ms stdev={stdev_ms:8.3f}ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
