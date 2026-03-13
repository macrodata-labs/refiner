from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import ArrowRowView
from ._lerobot_stats import _feature_stats


def collect_episode_tasks(*, tasks_raw: Any, task_raw: Any) -> list[str]:
    """Extract a stable ordered list of episode tasks.

    Keep duplicates removed and preserve encounter order; callers can then map to
    deterministic local task indexes.
    """

    tasks: list[str] = []
    if isinstance(tasks_raw, list):
        for value in tasks_raw:
            if isinstance(value, str) and value:
                tasks.append(value)

    if isinstance(task_raw, str) and task_raw and task_raw not in tasks:
        tasks.append(task_raw)
    return tasks


def resolve_task_index(
    task_to_index: dict[str, int],
    task_order: list[str],
    task: str,
) -> int:
    existing = task_to_index.get(task)
    if existing is not None:
        return existing

    next_index = len(task_order)
    task_to_index[task] = next_index
    task_order.append(task)
    return next_index


def infer_features(
    *,
    features: dict[str, dict[str, Any]],
    row: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
) -> None:
    """Populate feature specs for row-level and first-frame observation data."""

    for key, value in row.items():
        if key in {
            "frames",
            "task",
            "tasks",
            "metadata",
            "episode_index",
            "__shard_id",
        }:
            continue
        if hasattr(value, "media"):
            features.setdefault(
                key,
                {"dtype": "video", "shape": [3, 1, 1], "names": None},
            )
            continue
        if (not isinstance(value, bool)) and isinstance(value, (int, float, np.number)):
            features.setdefault(
                key,
                {"dtype": "float64", "shape": [1], "names": None},
            )

    if frames:
        sample = frames[0]
        for key, value in sample.items():
            if key in {"index", "episode_index", "task_index", "__shard_id"}:
                features.setdefault(
                    key,
                    {"dtype": "int64", "shape": [1], "names": None},
                )
                continue
            dtype = _infer_value_dtype(value)
            if dtype is None:
                continue
            arr = np.asarray(value)
            shape = [1] if arr.ndim == 0 else [int(x) for x in arr.shape]
            features.setdefault(
                key,
                {"dtype": dtype, "shape": shape, "names": None},
            )

    defaults = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key, spec in defaults.items():
        features.setdefault(key, spec)


def _infer_value_dtype(value: Any) -> str | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    if not isinstance(value, np.ndarray):
        try:
            value = np.asarray(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, np.ndarray):
        return None

    if value.dtype.kind in {"i", "u"}:
        return "int64"
    if value.dtype.kind == "f":
        return "float64"
    return None


def _arrow_frame_table(
    *,
    frames: list[Mapping[str, Any]],
    episode_index: int,
    start_index: int,
    default_task_idx: int | None,
) -> pa.Table | None:
    if not frames or not all(isinstance(frame, ArrowRowView) for frame in frames):
        return None

    first = frames[0]
    if not isinstance(first, ArrowRowView):
        return None

    names = tuple(name for name in first.names if name != "__shard_id")
    index_by_name = first.index_by_name
    columns = first.columns
    row_indices = [frame.row_idx for frame in frames if isinstance(frame, ArrowRowView)]
    if len(row_indices) != len(frames):
        return None
    if any(
        not isinstance(frame, ArrowRowView)
        or frame.names != first.names
        or frame.columns != columns
        for frame in frames
    ):
        return None

    take_idx = pa.array(row_indices, type=pa.int64())
    data = {
        name: columns[index_by_name[name]].take(take_idx)  # type: ignore[call-arg]
        for name in names
    }
    table = pa.table(data)
    row_count = len(frames)

    episode_col = pa.array([episode_index] * row_count, type=pa.int64())
    if "episode_index" in table.schema.names:
        idx = table.schema.get_field_index("episode_index")
        table = table.set_column(idx, "episode_index", episode_col)
    else:
        table = table.append_column("episode_index", episode_col)

    index_col = pa.array(
        [start_index + i for i in range(row_count)],
        type=pa.int64(),
    )
    if "index" in table.schema.names:
        idx = table.schema.get_field_index("index")
        table = table.set_column(idx, "index", index_col)
    else:
        table = table.append_column("index", index_col)

    if "frame_index" not in table.schema.names:
        table = table.append_column(
            "frame_index",
            pa.array(list(range(row_count)), type=pa.int64()),
        )
    if default_task_idx is not None and "task_index" not in table.schema.names:
        table = table.append_column(
            "task_index",
            pa.array([default_task_idx] * row_count, type=pa.int64()),
        )

    return table


def _to_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, bool) or value is None:
        return None

    arr = np.asarray(value)
    if not isinstance(arr, np.ndarray):
        return None
    if arr.dtype.kind not in {"i", "u", "f"}:
        return None
    if arr.size == 0:
        return None
    if not np.isfinite(arr.astype(np.float64, copy=False)).all():
        return None
    return arr.astype(np.float64, copy=False)


def compute_episode_stats(
    *,
    frames: list[Mapping[str, Any]],
    video_stats: Mapping[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    stats = _compute_frame_stats(frames)
    if video_stats:
        for key, vstats in video_stats.items():
            stats[key] = vstats
    return stats


def _compute_frame_stats(
    frames: list[Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    if not frames:
        return {}

    table = _frames_to_table(frames)
    if table.num_rows <= 0:
        return {}

    excluded = {"index", "episode_index", "task_index"}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in table.column_names:
        if key in excluded:
            continue

        column = table.column(key)
        if (
            pa.types.is_integer(column.type) or pa.types.is_floating(column.type)
        ) and column.null_count == 0:
            numeric = np.asarray(
                column.to_numpy(zero_copy_only=False), dtype=np.float64
            )
            if numeric.size == 0 or not np.isfinite(numeric).all():
                continue
            out[key] = _feature_stats(numeric, keepdims=True)
            continue

        list_numeric = _column_list_numeric_array(column)
        if list_numeric is not None:
            out[key] = _feature_stats(list_numeric, keepdims=False)
            continue

        values: list[np.ndarray] = []
        for value in column.to_pylist():
            arr = _to_numeric_array(value)
            if arr is None:
                continue
            values.append(arr)

        if not values:
            continue

        first_shape = values[0].shape
        if any(v.shape != first_shape for v in values):
            continue
        stacked = np.stack(values, axis=0)
        keepdims = stacked.ndim == 1
        out[key] = _feature_stats(stacked, keepdims=keepdims)

    return out


def _column_list_numeric_array(
    column: pa.ChunkedArray,
) -> np.ndarray | None:
    if column.null_count > 0:
        return None

    try:
        combined = column.combine_chunks()
    except (ValueError, TypeError):
        return None

    fixed_size: int | None = None
    value_type: pa.DataType | None = None
    if pa.types.is_fixed_size_list(combined.type):
        fixed_size = int(combined.type.list_size)
        value_type = combined.type.value_type
    elif pa.types.is_list(combined.type) or pa.types.is_large_list(combined.type):
        value_type = combined.type.value_type
    else:
        return None

    if value_type is None:
        return None
    if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
        return None

    values_array = combined.values
    if values_array.null_count > 0:
        return None

    values = np.asarray(values_array.to_numpy(zero_copy_only=False), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return None

    row_count = int(len(combined))
    if row_count <= 0:
        return None

    if fixed_size is not None:
        if fixed_size <= 0:
            return None
        if values.size != row_count * fixed_size:
            return None
        return values.reshape(row_count, fixed_size)

    offsets = np.asarray(
        combined.offsets.to_numpy(zero_copy_only=False), dtype=np.int64
    )
    if offsets.size != row_count + 1:
        return None
    lengths = np.diff(offsets)
    if lengths.size == 0:
        return None
    first = int(lengths[0])
    if first <= 0 or np.any(lengths != first):
        return None
    if values.size != row_count * first:
        return None
    return values.reshape(row_count, first)


def _frames_to_table(frames: Sequence[Mapping[str, Any]]) -> pa.Table:
    if not frames:
        return pa.table({})

    first = frames[0]
    if not isinstance(first, ArrowRowView):
        return pa.Table.from_pylist([dict(frame) for frame in frames])

    if not all(
        isinstance(frame, ArrowRowView)
        and frame.names == first.names
        and frame.columns == first.columns
        for frame in frames
    ):
        return pa.Table.from_pylist([dict(frame) for frame in frames])

    names = tuple(name for name in first.names if name != "__shard_id")
    row_indices = [frame.row_idx for frame in frames if isinstance(frame, ArrowRowView)]
    index_by_name = first.index_by_name
    columns = first.columns
    take_idx = pa.array(row_indices, type=pa.int64())
    data = {
        name: columns[index_by_name[name]].take(take_idx)  # type: ignore[call-arg]
        for name in names
    }
    return pa.table(data)
