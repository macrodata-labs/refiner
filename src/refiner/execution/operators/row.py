from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import cast

from refiner.pipeline.steps import (
    AsyncRowStep,
    BatchStep,
    FilterRowStep,
    FlatMapStep,
    RefinerStep,
    RowStep,
    normalize_batch_item,
    normalize_row_result,
)
from refiner.execution.asyncio.window import AsyncWindow
from refiner.execution.buffer import RowBuffer
from refiner.pipeline.data.row import Row

ShardDeltaFn = Callable[[dict[str, int]], None]


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
    scratch: list[list[Row]] = [[] for _ in ordered]
    async_windows: list[AsyncWindow[Row] | None] = [
        AsyncWindow[Row](
            max_in_flight=step.max_in_flight,
            preserve_order=step.preserve_order,
        )
        if isinstance(step, AsyncRowStep)
        else None
        for step in ordered
    ]

    async def _run_async_step(*, step: AsyncRowStep, row: Row) -> Row:
        result = step.apply_row_async(row)
        if inspect.isawaitable(result):
            result = await result
        # TODO (Hynek): Resolve the typing properly.
        return normalize_row_result(row, cast("Row | dict[str, object]", result))

    def _delta_add(delta: dict[str, int], shard_id: str, amount: int) -> None:
        if amount == 0:
            return
        next_value = delta.get(shard_id, 0) + amount
        if next_value == 0:
            delta.pop(shard_id, None)
        else:
            delta[shard_id] = next_value

    def _emit_delta(delta: dict[str, int] | None) -> None:
        if delta and on_shard_delta is not None:
            on_shard_delta(delta)

    def _delta_remove_rows(delta: dict[str, int] | None, rows: Iterable[Row]) -> None:
        if delta is None:
            return
        for row in rows:
            _delta_add(delta, row.require_shard_id(), -1)

    def _run_step(i: int, *, flush_all: bool) -> None:
        step = ordered[i]
        inp = queues[i]
        if not inp and not isinstance(step, AsyncRowStep):
            return
        out = queues[i + 1]

        if isinstance(step, RowStep):
            for row in inp.take_all():
                normalized = normalize_row_result(row, step.apply_row(row))
                out.append(normalized)
            return

        if isinstance(step, AsyncRowStep):
            window = async_windows[i]
            if window is None:
                return
            tmp = scratch[i]
            tmp.clear()
            for row in inp.take_all():
                window.submit_blocking(_run_async_step(step=step, row=row))
            tmp.extend(window.poll())
            if flush_all:
                tmp.extend(window.flush())
            if tmp:
                out.extend(tmp)
            return

        if isinstance(step, FilterRowStep):
            delta = {} if on_shard_delta is not None else None
            for row in inp.take_all():
                if step.apply_predicate(row):
                    out.append(row)
                elif delta is not None:
                    _delta_add(delta, row.require_shard_id(), -1)
            _emit_delta(delta)
            return

        if isinstance(step, FlatMapStep):
            tmp = scratch[i]
            tmp.clear()
            delta = {} if on_shard_delta is not None else None
            for row in inp.take_all():
                _delta_remove_rows(delta, (row,))
                for item in step.apply_row_many(row):
                    normalized = normalize_batch_item(item)
                    if normalized is not None:
                        if delta is not None:
                            _delta_add(delta, normalized.require_shard_id(), 1)
                        tmp.append(normalized)
            out.extend(tmp)
            _emit_delta(delta)
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
            tmp = scratch[i]
            tmp.clear()
            delta = {} if on_shard_delta is not None else None
            _delta_remove_rows(delta, batch_in)
            for item in step.apply_batch(batch_in):
                normalized = normalize_batch_item(item)
                if normalized is not None:
                    if delta is not None:
                        _delta_add(delta, normalized.require_shard_id(), 1)
                    tmp.append(normalized)
            out.extend(tmp)
            _emit_delta(delta)
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
