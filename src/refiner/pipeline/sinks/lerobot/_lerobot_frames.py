from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.lerobot._lerobot_stats import _feature_stats


def frame_table(
    *,
    frames: Sequence[Mapping[str, Any]] | Tabular,
    episode_index: int,
    start_index: int,
    task_index: int | None,
) -> pa.Table:
    table = frames.table if isinstance(frames, Tabular) else _rows_to_table(frames)
    row_count = int(table.num_rows)

    table = _set_or_append(
        table,
        "episode_index",
        pa.array([episode_index] * row_count, type=pa.int64()),
    )
    table = _set_or_append(
        table,
        "index",
        pa.array([start_index + i for i in range(row_count)], type=pa.int64()),
    )

    if "frame_index" not in table.schema.names:
        table = table.append_column(
            "frame_index",
            pa.array(range(row_count), type=pa.int64()),
        )
    if task_index is not None and "task_index" not in table.schema.names:
        table = table.append_column(
            "task_index",
            pa.array([task_index] * row_count, type=pa.int64()),
        )
    return table


def compute_episode_stats(
    *,
    frames: Sequence[Mapping[str, Any]] | Tabular,
    video_stats: Mapping[str, dict[str, np.ndarray]] | None = None,
    quantile_bins: int = 5000,
) -> dict[str, dict[str, np.ndarray]]:
    stats = _frame_stats(frames, quantile_bins=quantile_bins)
    if video_stats:
        stats.update(video_stats)
    return stats


def _frame_stats(
    frames: Sequence[Mapping[str, Any]] | Tabular,
    *,
    quantile_bins: int = 5000,
) -> dict[str, dict[str, np.ndarray]]:
    if isinstance(frames, Tabular):
        if frames.num_rows <= 0:
            return {}
    elif not frames:
        return {}

    table = frames.table if isinstance(frames, Tabular) else _rows_to_table(frames)
    if table.num_rows <= 0:
        return {}

    out: dict[str, dict[str, np.ndarray]] = {}
    for key in table.column_names:
        if key in {"index", "episode_index", "task_index"}:
            continue

        column = table.column(key)
        numeric = _numeric_column(column)
        if numeric is None:
            numeric = _stack_numeric_values(column.to_pylist())
        if numeric is None:
            continue
        out[key] = _feature_stats(
            numeric,
            num_quantile_bins=quantile_bins,
        )
    return out


def _rows_to_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({})
    if all(isinstance(row, Row) for row in rows):
        return Tabular.from_rows(cast(Sequence[Row], rows)).table
    return pa.Table.from_pylist([dict(row) for row in rows])


def _set_or_append(table: pa.Table, key: str, column: pa.Array) -> pa.Table:
    if key in table.schema.names:
        return table.set_column(table.schema.get_field_index(key), key, column)
    return table.append_column(key, column)


def _numeric_column(column: pa.ChunkedArray) -> np.ndarray | None:
    if column.null_count > 0:
        return None

    if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
        values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values if values.size and np.isfinite(values).all() else None

    try:
        combined = column.combine_chunks()
    except (TypeError, ValueError):
        return None

    if pa.types.is_fixed_size_list(combined.type):
        width = int(combined.type.list_size)
    elif pa.types.is_list(combined.type) or pa.types.is_large_list(combined.type):
        offsets = np.asarray(combined.offsets.to_numpy(zero_copy_only=False))
        lengths = np.diff(offsets)
        if lengths.size == 0 or np.any(lengths != lengths[0]) or lengths[0] <= 0:
            return None
        width = int(lengths[0])
    else:
        return None

    value_type = combined.type.value_type
    if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
        return None
    if combined.values.null_count > 0:
        return None

    values = np.asarray(
        combined.values.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if values.size == 0 or not np.isfinite(values).all():
        return None
    row_count = len(combined)
    return (
        values.reshape(row_count, width) if values.size == row_count * width else None
    )


def _stack_numeric_values(values: list[Any]) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for value in values:
        if value is None or isinstance(value, bool):
            return None
        array = np.asarray(value)
        if array.size == 0 or array.dtype.kind not in {"i", "u", "f"}:
            return None
        numeric = array.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            return None
        arrays.append(numeric)

    if not arrays:
        return None
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        return None
    return np.stack(arrays, axis=0)
