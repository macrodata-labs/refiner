from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from refiner.processors.step import (
    BatchStep,
    FlatMapStep,
    RefinerStep,
    RowStep,
    normalize_batch_item,
    normalize_row_result,
)
from refiner.runtime.execution.row_queue import RowQueue
from refiner.sources.row import Row


def execute_row_steps(
    rows: Iterable[Row], steps: Sequence[RefinerStep]
) -> Iterator[Row]:
    """Execute row/batch/flatmap steps using per-step queues.

    This preserves existing batch semantics (including cross-input batch accumulation)
    and is used for Python-UDF row segments in the segmented executor.
    """
    ordered = tuple(steps)
    if not ordered:
        yield from rows
        return

    queues: list[RowQueue] = [RowQueue() for _ in range(len(ordered) + 1)]
    scratch: list[list[Row]] = [[] for _ in ordered]

    def _run_step(i: int, *, flush_all: bool) -> None:
        step = ordered[i]
        inp = queues[i]
        if not inp:
            return
        out = queues[i + 1]

        if isinstance(step, RowStep):
            for row in inp.take_all():
                normalized = normalize_row_result(row, step.apply_row(row))
                out.append(normalized)
            return

        if isinstance(step, FlatMapStep):
            tmp = scratch[i]
            tmp.clear()
            for row in inp.take_all():
                for item in step.apply_row_many(row):
                    normalized = normalize_batch_item(item)
                    if normalized is not None:
                        tmp.append(normalized)
            out.extend(tmp)
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
            for item in step.apply_batch(batch_in):
                normalized = normalize_batch_item(item)
                if normalized is not None:
                    tmp.append(normalized)
            out.extend(tmp)
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


__all__ = ["execute_row_steps"]
