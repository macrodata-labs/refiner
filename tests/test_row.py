from __future__ import annotations

import pyarrow as pa
import pytest

from refiner.runtime.metrics_context import (
    set_active_step_index,
    set_active_user_metrics_emitter,
)
from refiner.runtime.execution.vectorized import iter_table_rows
from refiner.sources.row import DictRow, Row


class _CaptureEmitter:
    def __init__(self) -> None:
        self.counters: list[dict] = []
        self.histograms: list[dict] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        del kwargs

    def emit_user_histogram(self, **kwargs) -> None:
        self.histograms.append(kwargs)

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def test_row_copy_preserves_internal_fields() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"})
    out = row.copy({"x": 2, "y": 3})
    assert out["x"] == 2
    assert out["y"] == 3
    assert out.shard_id == "s-1"
    assert "__shard_id" not in out


def test_row_from_row_with_row_base() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"})
    out = Row.from_row(row, {"z": 9})
    assert out["x"] == 1
    assert out["z"] == 9
    assert out.shard_id == "s-1"
    assert "__shard_id" not in out


def test_row_from_row_with_mapping_base() -> None:
    base = {"x": 1, "__shard_id": "s-1"}
    out = Row.from_row(base, {"y": 2})
    assert out["x"] == 1
    assert out["y"] == 2
    assert out.shard_id == "s-1"
    assert "__shard_id" not in out


def test_dict_row_infers_shard_id_field() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"})
    assert row.shard_id == "s-1"
    assert row.require_shard_id() == "s-1"
    assert "__shard_id" not in row


def test_with_shard_id_sets_field_only() -> None:
    row = DictRow({"x": 1}).with_shard_id("s-2")
    assert row.shard_id == "s-2"
    assert row.require_shard_id() == "s-2"
    assert "__shard_id" not in row


def test_drop_internal_key_keeps_tracking_field() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"}).drop("__shard_id")
    assert row.shard_id == "s-1"
    assert row.require_shard_id() == "s-1"
    assert "__shard_id" not in row


def test_update_can_override_shard_id() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"}).update({"__shard_id": "s-2"})
    assert row.shard_id == "s-2"
    assert row.require_shard_id() == "s-2"
    assert "__shard_id" not in row


def test_iter_table_rows_populates_shard_id_field() -> None:
    table = pa.table({"x": [1, 2], "__shard_id": ["a", "b"]})
    rows = list(iter_table_rows(table))
    assert [row.shard_id for row in rows] == ["a", "b"]
    assert [row.require_shard_id() for row in rows] == ["a", "b"]
    assert "__shard_id" not in rows[0]
    with pytest.raises(KeyError):
        _ = rows[0]["__shard_id"]


def test_internal_shard_key_not_counted_in_len_or_iter() -> None:
    row = DictRow({"x": 1, "__shard_id": "s-1"})
    assert list(row) == ["x"]
    assert len(row) == 1


def test_row_log_throughput_uses_row_shard_id() -> None:
    emitter = _CaptureEmitter()
    row = DictRow({"x": 1}).with_shard_id("s-1")
    with set_active_user_metrics_emitter(emitter), set_active_step_index(7):
        row.log_throughput("rows_processed", 3, unit="rows")

    assert emitter.counters == [
        {
            "label": "rows_processed",
            "value": 3.0,
            "shard_id": "s-1",
            "step_index": 7,
            "unit": "rows",
        }
    ]


def test_row_log_histogram_uses_row_shard_id() -> None:
    emitter = _CaptureEmitter()
    row = DictRow({"x": 1}).with_shard_id("s-2")
    with set_active_user_metrics_emitter(emitter), set_active_step_index(3):
        row.log_histogram("latency", 12, per="request", unit="ms")

    assert emitter.histograms == [
        {
            "label": "latency",
            "value": 12.0,
            "shard_id": "s-2",
            "per": "request",
            "step_index": 3,
            "unit": "ms",
        }
    ]


def test_row_log_methods_require_shard_id() -> None:
    row = DictRow({"x": 1})
    with pytest.raises(ValueError, match="missing shard_id"):
        row.log_throughput("x", 1)
    with pytest.raises(ValueError, match="missing shard_id"):
        row.log_histogram("x", 1)
