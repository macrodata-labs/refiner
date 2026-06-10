from __future__ import annotations

import asyncio
from typing import Any

import pytest

from refiner.execution.operators.row import execute_row_steps
from refiner.pipeline import from_items
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.steps import FnAsyncRowStep
from refiner.worker.context import set_active_run_context
from refiner.worker.lifecycle import FinalizedShardWorker
from refiner.worker.metrics.emitter import UserMetricsEmitter


class _RecordingTelemetryEmitter(UserMetricsEmitter):
    def __init__(self) -> None:
        self.counters: list[dict[str, Any]] = []

    def emit_user_counter(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        self.counters.append(
            {
                "label": label,
                "value": value,
                "shard_id": shard_id,
                "step_index": step_index,
                "unit": unit,
            }
        )

    def emit_user_gauge(
        self,
        *,
        label: str,
        value: float,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, kind, step_index, unit

    def register_user_gauge(
        self,
        *,
        label: str,
        callback: Any,
        kind: str | None,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, callback, kind, step_index, unit

    def emit_user_histogram(
        self,
        *,
        label: str,
        value: float,
        shard_id: str,
        per: str,
        step_index: int | None,
        unit: str | None,
    ) -> None:
        del label, value, shard_id, per, step_index, unit


class _RuntimeLifecycle:
    def claim(self, previous: Shard | None = None) -> Shard | None:
        del previous
        return None

    def heartbeat(self, shards: list[Shard]) -> None:
        del shards

    def complete(self, shard: Shard) -> None:
        del shard

    def fail(self, shard: Shard, error: str | None = None) -> None:
        del shard, error

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        del stage_index
        return []


async def _delayed_value(value: int, delay: float) -> dict[str, int]:
    await asyncio.sleep(delay)
    return {"x": value}


async def _failing_async_row(row: Row) -> dict[str, int]:
    del row
    await asyncio.sleep(0)
    raise RuntimeError("async row failed")


def test_map_async_preserves_order_by_default() -> None:
    pipeline = from_items([1, 2, 3]).map_async(
        lambda row: _delayed_value(
            int(row["item"]), 0.03 if int(row["item"]) == 1 else 0.0
        )
    )

    out = list(pipeline.iter_rows())
    assert [int(row["x"]) for row in out] == [1, 2, 3]


def test_map_async_without_order_preservation_still_emits_all_rows() -> None:
    pipeline = from_items([1, 2, 3]).map_async(
        lambda row: _delayed_value(
            int(row["item"]), 0.03 if int(row["item"]) == 1 else 0.0
        ),
        preserve_order=False,
    )

    out = list(pipeline.iter_rows())
    assert sorted(int(row["x"]) for row in out) == [1, 2, 3]


def test_map_async_counts_rows_processed_only_after_completion() -> None:
    emitter = _RecordingTelemetryEmitter()
    step = FnAsyncRowStep(
        fn=_failing_async_row,
        index=7,
        max_in_flight=1,
        op_name="map_async",
    )
    rows = [DictRow({"item": 1}, shard_id="shard-1")]

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker",
        worker_name=None,
        runtime_lifecycle=_RuntimeLifecycle(),
        user_metrics_emitter=emitter,
    ):
        with pytest.raises(RuntimeError, match="async row failed"):
            list(execute_row_steps(rows, [step]))

    assert [
        metric for metric in emitter.counters if metric["label"] == "rows_processed"
    ] == []


def test_map_async_closes_async_callable_after_iteration() -> None:
    class AsyncCallable:
        def __init__(self) -> None:
            self.closed = False

        async def __call__(self, row: Row) -> dict[str, int]:
            return {"x": int(row["item"])}

        async def aclose(self) -> None:
            self.closed = True

    fn = AsyncCallable()
    pipeline = from_items([1, 2, 3]).map_async(fn)

    out = list(pipeline.iter_rows())

    assert [int(row["x"]) for row in out] == [1, 2, 3]
    assert fn.closed is True
