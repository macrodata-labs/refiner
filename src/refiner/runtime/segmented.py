from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa

from refiner.processors.step import RefinerStep, VectorizedOp, VectorizedSegmentStep
from refiner.runtime.row_steps import execute_row_steps
from refiner.runtime.vectorized import (
    apply_vectorized_op,
    chunk_rows,
    rows_to_table,
    table_to_rows,
)
from refiner.sources.row import Row

_DEFAULT_VECTORIZED_CHUNK_ROWS = 2048


@dataclass(frozen=True, slots=True)
class RowSegment:
    steps: tuple[RefinerStep, ...]


@dataclass(frozen=True, slots=True)
class VectorSegment:
    ops: tuple[VectorizedOp, ...]


Segment = RowSegment | VectorSegment
StreamItem = Row | list[Row] | pa.Table | pa.RecordBatch


def compile_segments(steps: Sequence[RefinerStep]) -> tuple[Segment, ...]:
    out: list[Segment] = []
    row_steps: list[RefinerStep] = []
    for step in steps:
        if isinstance(step, VectorizedSegmentStep):
            if row_steps:
                out.append(RowSegment(steps=tuple(row_steps)))
                row_steps.clear()
            out.append(VectorSegment(ops=step.ops))
            continue
        row_steps.append(step)
    if row_steps:
        out.append(RowSegment(steps=tuple(row_steps)))
    return tuple(out)


def execute_segments(
    stream: Iterable[StreamItem],
    segments: Sequence[Segment],
    *,
    vectorized_chunk_rows: int = _DEFAULT_VECTORIZED_CHUNK_ROWS,
) -> Iterator[Row]:
    current: Iterable[StreamItem] = stream
    for segment in segments:
        if isinstance(segment, VectorSegment):
            current = _execute_vector_segment(
                current,
                segment.ops,
                vectorized_chunk_rows=vectorized_chunk_rows,
            )
        else:
            current = execute_row_steps(iter_rows(current), segment.steps)
    yield from iter_rows(current)


def iter_rows(stream: Iterable[StreamItem]) -> Iterator[Row]:
    for item in stream:
        if isinstance(item, Row):
            yield item
            continue
        if isinstance(item, list):
            for row in item:
                yield row
            continue
        if isinstance(item, pa.RecordBatch):
            yield from table_to_rows(pa.Table.from_batches([item]))
            continue
        if isinstance(item, pa.Table):
            yield from table_to_rows(item)
            continue
        raise TypeError(f"Unsupported stream item: {type(item)!r}")


def _execute_vector_segment(
    stream: Iterable[StreamItem],
    ops: Sequence[VectorizedOp],
    *,
    vectorized_chunk_rows: int,
) -> Iterator[Row]:
    pending_rows: list[Row] = []

    def _run_table(table: pa.Table) -> Iterator[Row]:
        out = table
        for op in ops:
            out = apply_vectorized_op(out, op)
        yield from table_to_rows(out)

    def _flush_rows() -> Iterator[Row]:
        if not pending_rows:
            return
        batches = chunk_rows(pending_rows, vectorized_chunk_rows)
        pending_rows.clear()
        for batch in batches:
            yield from _run_table(rows_to_table(batch))

    for item in stream:
        if isinstance(item, Row):
            pending_rows.append(item)
            if len(pending_rows) >= vectorized_chunk_rows:
                yield from _flush_rows()
            continue

        if isinstance(item, list):
            pending_rows.extend(item)
            while len(pending_rows) >= vectorized_chunk_rows:
                chunk = pending_rows[:vectorized_chunk_rows]
                del pending_rows[:vectorized_chunk_rows]
                yield from _run_table(rows_to_table(chunk))
            continue

        if isinstance(item, pa.RecordBatch):
            yield from _flush_rows()
            yield from _run_table(pa.Table.from_batches([item]))
            continue

        if isinstance(item, pa.Table):
            yield from _flush_rows()
            yield from _run_table(item)
            continue

        raise TypeError(f"Unsupported stream item: {type(item)!r}")

    yield from _flush_rows()


__all__ = [
    "compile_segments",
    "execute_segments",
    "iter_rows",
    "RowSegment",
    "VectorSegment",
]
