from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator, Sequence
from typing import cast

from refiner.execution.asyncio.window import AsyncWindow
from refiner.execution.buffer import RowBuffer
from refiner.execution.tracking.shards import ShardDeltaFn, ShardDeltaTracker
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import (
    AsyncRowStep,
    BatchStep,
    FilterRowStep,
    FlatMapStep,
    MapResult,
    RefinerStep,
    RowStep,
)
from refiner.worker.context import set_active_step_index
from refiner.worker.metrics.api import register_gauge


def execute_row_steps(
    rows: Iterable[Row],
    steps: Sequence[RefinerStep],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
) -> Iterator[Row]:
    """Execute row/batch/flatmap steps using per-step queues.

    This preserves existing batch semantics (including cross-input batch accumulation)
    and is used for Python-UDF row segments in the segmented executor.
    """
    ordered = tuple(steps)
    if not ordered:
        yield from rows
        return

    queues: list[RowBuffer] = [RowBuffer() for _ in range(len(ordered) + 1)]
    async_windows: list[AsyncWindow[Row] | None] = [
        AsyncWindow[Row](
            max_in_flight=step.max_in_flight,
            preserve_order=step.preserve_order,
        )
        if isinstance(step, AsyncRowStep)
        else None
        for step in ordered
    ]
    for i, step in enumerate(ordered):
        window = async_windows[i]
        if window is not None:
            register_gauge(
                "in_flight",
                lambda window=window: len(window),
                unit="rows",
                step_index=step.index,
            )

    async def _run_async_step(*, step: AsyncRowStep, row: Row) -> Row:
        with set_active_step_index(step.index):
            result = step.apply_row_async(row)
            if inspect.isawaitable(result):
                result = await result
            result = cast(MapResult, result)
            if isinstance(result, Row):
                return result
            if isinstance(result, dict):
                return row.update(result)
            raise TypeError(f"Unsupported map_async() result type: {type(result)!r}")

    def _run_step(i: int, *, flush_all: bool) -> None:
        step = ordered[i]
        inp = queues[i]
        if not inp and not isinstance(step, AsyncRowStep):
            return
        out = queues[i + 1]
        with set_active_step_index(step.index):
            if isinstance(step, RowStep):
                for row in inp.take_all():
                    row.log_throughput("rows_processed", 1, unit="rows")
                    result = step.apply_row(row)
                    if isinstance(result, Row):
                        out.append(result)
                    elif isinstance(result, dict):
                        out.append(row.update(result))
                    else:
                        raise TypeError(
                            f"Unsupported map() result type: {type(result)!r}"
                        )
                return

            if isinstance(step, AsyncRowStep):
                window = async_windows[i]
                if window is None:
                    return
                for row in inp.take_all():
                    row.log_throughput("rows_processed", 1, unit="rows")
                    window.submit_blocking(_run_async_step(step=step, row=row))
                out.extend(window.take_completed())
                if flush_all:
                    out.extend(window.drain())
                return

            if isinstance(step, FilterRowStep):
                with ShardDeltaTracker(on_shard_delta) as delta:
                    for row in inp.take_all():
                        row.log_throughput("rows_processed", 1, unit="rows")
                        if step.apply_predicate(row):
                            row.log_throughput("rows_kept", 1, unit="rows")
                            out.append(row)
                        else:
                            row.log_throughput("rows_dropped", 1, unit="rows")
                            if row.shard_id is not None:
                                delta.add(row.shard_id, -1)
                return

            if isinstance(step, FlatMapStep):
                with ShardDeltaTracker(on_shard_delta) as delta:
                    for row in inp.take_all():
                        produced = 0
                        emitted_by_shard: dict[str, int] = {}
                        for item in step.apply_row_many(row):
                            if isinstance(item, Row):
                                emitted = item
                            elif isinstance(item, dict):
                                emitted = row.update(item)
                            else:
                                raise TypeError(
                                    f"Unsupported flat_map result type: {type(item)!r}"
                                )
                            produced += 1
                            if emitted.shard_id is not None:
                                emitted_by_shard[emitted.shard_id] = (
                                    emitted_by_shard.get(emitted.shard_id, 0) + 1
                                )
                            out.append(emitted)
                        if row.shard_id is not None:
                            delta.add(row.shard_id, -1)
                        for shard_id, count in emitted_by_shard.items():
                            delta.add(shard_id, count)
                        row.log_histogram(
                            "rows_out", produced, unit="rows", per="input_row"
                        )
                return

            if isinstance(step, BatchStep):
                if flush_all:
                    batch_in = inp.take_all()
                else:
                    n = (len(inp) // step.batch_size) * step.batch_size
                    if n == 0:
                        return
                    batch_in = inp.take(n)
                if not batch_in:
                    return
                with ShardDeltaTracker(on_shard_delta) as delta:
                    delta.remove_rows(batch_in)
                    for item in step.apply_batch(batch_in):
                        item.log_throughput("rows_out", 1, unit="rows")
                        if item.shard_id is not None:
                            delta.add(item.shard_id, 1)
                        out.append(item)
                return

            raise TypeError(f"Unsupported row-segment step: {type(step)!r}")

    def _pump(flush_all: bool) -> None:
        for i in range(len(ordered)):
            _run_step(i, flush_all=flush_all)

    def _drain_output() -> Iterator[Row]:
        outq = queues[-1]
        if not outq:
            return
        yield from outq.take_all()

    for row in rows:
        queues[0].append(row)
        _pump(flush_all=False)
        yield from _drain_output()

    _pump(flush_all=True)
    yield from _drain_output()


__all__ = ["execute_row_steps", "ShardDeltaFn"]
