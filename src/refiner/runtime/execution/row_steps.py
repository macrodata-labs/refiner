from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Sequence

from refiner.processors.step import (
    AsyncRowStep,
    BatchStep,
    FilterRowStep,
    FlatMapStep,
    RefinerStep,
    RowStep,
    normalize_batch_item,
    normalize_row_result,
)
from refiner.runtime.execution.async_window import AsyncWindow
from refiner.runtime.execution.row_queue import RowQueue
from refiner.sources.row import Row

ShardDeltaFn = Callable[[dict[str, int]], None]


def execute_row_steps(
    rows: Iterable[Row],
    steps: Sequence[RefinerStep],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
) -> Iterator[Row]:
    # Shard tracking contract:
    # - The source seeds each row with an internal shard id.
    # - Steps that can change cardinality emit per-shard deltas immediately.
    # - 1:1 steps do not emit deltas because net live-row count is unchanged.
    ordered = tuple(steps)
    if not ordered:
        yield from rows
        return

    track = on_shard_delta is not None
    queues: list[RowQueue] = [RowQueue() for _ in range(len(ordered) + 1)]
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
        return normalize_row_result(row, result)

    def _delta_add(delta: dict[str, int], shard: str, amount: int) -> None:
        # Keep delta maps sparse: drop zero entries eagerly.
        if amount == 0:
            return
        next_value = delta.get(shard, 0) + amount
        if next_value == 0:
            delta.pop(shard, None)
            return
        delta[shard] = next_value

    def _emit_delta(delta: dict[str, int]) -> None:
        # Emit only non-empty updates to avoid callback churn.
        if on_shard_delta is None:
            return
        if delta:
            on_shard_delta(delta)

    def _run_step(i: int, *, flush_all: bool) -> None:
        step = ordered[i]
        inp = queues[i]
        out = queues[i + 1]
        if not inp and not isinstance(step, AsyncRowStep):
            return

        if isinstance(step, RowStep):
            # 1:1 map.
            # Consumes exactly one row and emits exactly one row.
            for row in inp.take_all():
                normalized = normalize_row_result(row, step.apply_row(row))
                out.append(normalized)
            return

        if isinstance(step, FilterRowStep):
            # 1:0/1 filter.
            # Every dropped row decrements the producing shard by 1.
            consumed_rows = inp.take_all()
            tmp = scratch[i]
            tmp.clear()
            delta: dict[str, int] | None = {} if track else None
            for row in consumed_rows:
                if step.apply_predicate(row):
                    tmp.append(row)
                    continue
                if delta is not None:
                    _delta_add(delta, row.require_shard_id(), -1)
            out.extend(tmp)
            if delta is not None:
                _emit_delta(delta)
            return

        if isinstance(step, AsyncRowStep):
            # 1:1 async map.
            # Cardinality is still 1:1; ordering may differ when preserve_order=False.
            window = async_windows[i]
            if window is None:
                return
            tmp = scratch[i]
            tmp.clear()
            while inp:
                cap = window.capacity
                if cap > 0:
                    for row in inp.take(cap):
                        window.submit(_run_async_step(step=step, row=row))
                tmp.extend(window.drain(flush=False))
            if flush_all:
                tmp.extend(window.drain(flush=True))
            if tmp:
                out.extend(tmp)
            return

        if isinstance(step, FlatMapStep):
            # 1:0..N flat_map.
            # Delta is produced - consumed per shard for this step execution.
            consumed_rows = inp.take_all()
            tmp = scratch[i]
            tmp.clear()
            delta: dict[str, int] | None = {} if track else None
            for row in consumed_rows:
                if delta is not None:
                    _delta_add(delta, row.require_shard_id(), -1)
                for item in step.apply_row_many(row):
                    normalized = normalize_batch_item(item)
                    if normalized is None:
                        continue
                    if delta is not None:
                        _delta_add(delta, normalized.require_shard_id(), 1)
                    tmp.append(normalized)
            out.extend(tmp)
            if delta is not None:
                _emit_delta(delta)
            return

        if isinstance(step, BatchStep):
            # N:0..M batch_map.
            # We consume input rows first, then add back outputs by their shard ids.
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

            delta: dict[str, int] | None = {} if track else None
            if delta is not None:
                for row in batch_in:
                    _delta_add(delta, row.require_shard_id(), -1)
            for item in step.apply_batch(batch_in):
                normalized = normalize_batch_item(item)
                if normalized is None:
                    continue
                if delta is not None:
                    _delta_add(delta, normalized.require_shard_id(), 1)
                tmp.append(normalized)
            out.extend(tmp)
            if delta is not None:
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
