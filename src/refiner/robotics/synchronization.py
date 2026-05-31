from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.tabular import Tabular

SyncMethod = Literal["nearest", "interpolate", "hold"]
TimestampedValue = tuple[int, Any]

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _AlignedValue:
    timestamp_ns: int
    value: Any
    skew_ns: int


def sparse_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[TimestampedValue]],
) -> pa.Table:
    values_by_field: dict[str, dict[int, list[Any]]] = {}
    for name, source in fields.items():
        values: dict[int, list[Any]] = defaultdict(list)
        for event in topic_events.get(source[0], ()):
            values[event[0]].append(source_value(event[1], source[1], default=None))
        values_by_field[name] = values
    timestamps = sorted(
        {
            event[0]
            for source in fields.values()
            for event in topic_events.get(source[0], ())
        }
    )
    rows: list[dict[str, Any]] = []
    for timestamp_ns in timestamps:
        repeats = max(
            (len(values.get(timestamp_ns, ())) for values in values_by_field.values()),
            default=1,
        )
        for duplicate_index in range(repeats):
            row: dict[str, Any] = {
                "frame_index": len(rows),
                "timestamp": timestamp_ns / 1e9,
            }
            for name, values in values_by_field.items():
                timestamp_values = values.get(timestamp_ns, ())
                row[name] = (
                    timestamp_values[duplicate_index]
                    if duplicate_index < len(timestamp_values)
                    else None
                )
            rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def aligned_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[TimestampedValue]],
    *,
    sync_primary_events: Sequence[TimestampedValue],
    sync_primary: tuple[str, str | None],
    sync_method: SyncMethod,
    include_skew: bool,
) -> pa.Table:
    sync_primary_timestamps = [event[0] for event in sync_primary_events]
    aligned_values = {
        name: align_values(
            topic_events.get(source[0], ()),
            sync_primary_timestamps,
            source[1],
            method=sync_method,
        )
        for name, source in fields.items()
        if source[0] != sync_primary[0]
    }
    rows: list[dict[str, Any]] = []
    for index, sync_primary_event in enumerate(sync_primary_events):
        timestamp_ns = sync_primary_event[0]
        row: dict[str, Any] = {
            "frame_index": index,
            "timestamp": timestamp_ns / 1e9,
        }
        for name in fields:
            source = fields[name]
            if source[0] == sync_primary[0]:
                row[name] = source_value(sync_primary_event[1], source[1], default=None)
                continue
            aligned = aligned_values[name][index]
            if aligned is None:
                row[name] = None
                continue
            row[name] = aligned.value
            if include_skew:
                row[f"mcap.{name}.timestamp"] = aligned.timestamp_ns / 1e9
                row[f"mcap.{name}.skew_ms"] = aligned.skew_ns / 1e6
        rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def source_value(value: Any, field_path: str | None, *, default: Any = _MISSING) -> Any:
    if field_path is None:
        return value
    current = value
    try:
        for part in field_path.split("."):
            if isinstance(current, Mapping):
                current = current[part]
            else:
                current = getattr(current, part)
    except (KeyError, AttributeError):
        if default is not _MISSING:
            return default
        raise
    return current


def align_values(
    events: Sequence[TimestampedValue],
    timestamps_ns: Sequence[int],
    field_path: str | None,
    *,
    method: SyncMethod,
) -> list[_AlignedValue | None]:
    if not events:
        return [None] * len(timestamps_ns)
    sorted_events = sorted(events, key=lambda event: event[0])
    source_timestamps = [event[0] for event in sorted_events]
    source_values = [
        source_value(event[1], field_path, default=None) for event in sorted_events
    ]
    align = _nearest_value
    if method == "hold":
        align = _hold_value
    elif method == "interpolate":
        align = _interpolate_value
    return [
        align(timestamp_ns, source_timestamps, source_values)
        for timestamp_ns in timestamps_ns
    ]


def _nearest_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue:
    index = bisect_left(source_timestamps, timestamp_ns)
    if index == 0:
        source_index = 0
    elif index == len(source_timestamps):
        source_index = len(source_timestamps) - 1
    else:
        left = source_timestamps[index - 1]
        right = source_timestamps[index]
        source_index = (
            index - 1 if timestamp_ns - left <= right - timestamp_ns else index
        )
    source_timestamp = int(source_timestamps[source_index])
    return _AlignedValue(
        timestamp_ns=source_timestamp,
        value=source_values[source_index],
        skew_ns=source_timestamp - timestamp_ns,
    )


def _hold_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue | None:
    index = bisect_right(source_timestamps, timestamp_ns) - 1
    if index < 0:
        return None
    source_timestamp = int(source_timestamps[index])
    return _AlignedValue(
        timestamp_ns=source_timestamp,
        value=source_values[index],
        skew_ns=source_timestamp - timestamp_ns,
    )


def _interpolate_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue:
    index = bisect_left(source_timestamps, timestamp_ns)
    if index == 0 or index == len(source_timestamps):
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    left_value = source_values[index - 1]
    right_value = source_values[index]
    left_array = np.asarray(left_value)
    right_array = np.asarray(right_value)
    if (
        left_array.dtype.kind not in "iufc"
        or right_array.dtype.kind not in "iufc"
        or left_array.shape != right_array.shape
    ):
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    left_timestamp = int(source_timestamps[index - 1])
    right_timestamp = int(source_timestamps[index])
    span = right_timestamp - left_timestamp
    if span <= 0:
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    alpha = (timestamp_ns - left_timestamp) / span
    value = (1.0 - alpha) * left_array + alpha * right_array
    if value.shape == ():
        value = value.item()
    else:
        value = value.tolist()

    nearest_timestamp = (
        left_timestamp
        if timestamp_ns - left_timestamp <= right_timestamp - timestamp_ns
        else right_timestamp
    )
    return _AlignedValue(
        timestamp_ns=nearest_timestamp,
        value=value,
        skew_ns=nearest_timestamp - timestamp_ns,
    )
