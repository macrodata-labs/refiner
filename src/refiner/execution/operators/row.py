from __future__ import annotations

import inspect
from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence
from typing import cast

from refiner.execution.asyncio.window import AsyncWindow
from refiner.execution.asyncio.runtime import submit
from refiner.execution.buffer import RowBuffer
from refiner.execution.tracking.shards import ShardDeltaFn, ShardDeltaTracker
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import (
    AsyncBatchStep,
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

AsyncCloseFn = Callable[[], Coroutine[object, object, None]]
AsyncWindowRegistry = dict[int, AsyncWindow[Row]]
AsyncBatchResult = tuple[list[Row], list[Row]]
AsyncBatchWindowRegistry = dict[int, AsyncWindow[AsyncBatchResult]]


class AsyncStepTeardownError(RuntimeError):
    """Raised when an async pipeline callable fails to close."""


def close_async_steps(steps: Sequence[RefinerStep]) -> None:
    """Close async callables owned by the provided pipeline steps."""
    close_fns: list[AsyncCloseFn] = []
    for step in steps:
        if not isinstance(step, AsyncRowStep | AsyncBatchStep):
            continue
        fn = getattr(step, "fn", None)
        close = getattr(fn, "aclose", None)
        if close is not None:
            close_fns.append(cast(AsyncCloseFn, close))
    first_error: Exception | None = None
    for close in close_fns:
        try:
            submit(close()).result()
        except Exception as e:
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise AsyncStepTeardownError(str(first_error)) from first_error


def execute_row_steps(
    rows: Iterable[Row],
    steps: Sequence[RefinerStep],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
    async_window_registry: AsyncWindowRegistry | None = None,
    async_batch_window_registry: AsyncBatchWindowRegistry | None = None,
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

    def _async_window(step: AsyncRowStep) -> AsyncWindow[Row]:
        window = (
            async_window_registry.get(step.index)
            if async_window_registry is not None
            else None
        )
        if window is None:
            window = AsyncWindow[Row](
                max_in_flight=step.max_in_flight,
                preserve_order=step.preserve_order,
            )
            if async_window_registry is not None:
                async_window_registry[step.index] = window
            register_gauge(
                "in_flight",
                lambda window=window: len(window),
                unit="rows",
                step_index=step.index,
            )
        return window

    def _async_batch_window(
        step: AsyncBatchStep,
    ) -> AsyncWindow[AsyncBatchResult]:
        window = (
            async_batch_window_registry.get(step.index)
            if async_batch_window_registry is not None
            else None
        )
        if window is None:
            window = AsyncWindow[AsyncBatchResult](
                max_in_flight=step.max_in_flight,
                preserve_order=step.preserve_order,
            )
            if async_batch_window_registry is not None:
                async_batch_window_registry[step.index] = window
            register_gauge(
                "in_flight_batches",
                lambda window=window: len(window),
                unit="batches",
                step_index=step.index,
            )
        return window

    async_windows = [
        _async_window(step) if isinstance(step, AsyncRowStep) else None
        for step in ordered
    ]
    async_batch_windows = [
        _async_batch_window(step) if isinstance(step, AsyncBatchStep) else None
        for step in ordered
    ]

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

    async def _run_async_batch(
        *, step: AsyncBatchStep, rows: list[Row]
    ) -> AsyncBatchResult:
        with set_active_step_index(step.index):
            result = step.apply_batch_async(rows)
            if inspect.isawaitable(result):
                result = await result
            output = list(cast(Iterable[Row], result))
            for item in output:
                if not isinstance(item, Row):
                    raise TypeError(
                        f"Unsupported batch_map_async() result type: {type(item)!r}"
                    )
            return rows, output

    def _run_step(i: int, *, flush_all: bool) -> None:
        step = ordered[i]
        inp = queues[i]
        if not inp and not isinstance(step, AsyncRowStep | AsyncBatchStep):
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
                    window.submit_blocking(_run_async_step(step=step, row=row))
                completed_rows = window.take_completed()
                for row in completed_rows:
                    row.log_throughput("rows_processed", 1, unit="rows")
                out.extend(completed_rows)
                if flush_all:
                    drained_rows = window.drain()
                    for row in drained_rows:
                        row.log_throughput("rows_processed", 1, unit="rows")
                    out.extend(drained_rows)
                return

            if isinstance(step, AsyncBatchStep):
                window = async_batch_windows[i]
                if window is None:
                    return

                def emit(completed: list[AsyncBatchResult]) -> None:
                    if not completed:
                        return
                    with ShardDeltaTracker(on_shard_delta) as delta:
                        for batch_in, batch_out in completed:
                            delta.remove_rows(batch_in)
                            for item in batch_out:
                                item.log_throughput("rows_out", 1, unit="rows")
                                if item.shard_id is not None:
                                    delta.add(item.shard_id, 1)
                                out.append(item)

                while len(inp) >= step.batch_size or (flush_all and inp):
                    size = step.batch_size if len(inp) >= step.batch_size else len(inp)
                    batch_in = inp.take(size)
                    window.submit_blocking(_run_async_batch(step=step, rows=batch_in))
                    emit(window.take_completed())
                if flush_all:
                    emit(window.drain())
                else:
                    emit(window.take_completed())
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

    try:
        for row in rows:
            queues[0].append(row)
            _pump(flush_all=False)
            yield from _drain_output()

        _pump(flush_all=True)
        yield from _drain_output()
    finally:
        for window in async_windows:
            if window is not None:
                window.cancel_pending()
        for window in async_batch_windows:
            if window is not None:
                window.cancel_pending()


__all__ = [
    "AsyncWindowRegistry",
    "AsyncBatchWindowRegistry",
    "AsyncStepTeardownError",
    "close_async_steps",
    "execute_row_steps",
    "ShardDeltaFn",
]
