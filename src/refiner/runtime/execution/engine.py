from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa

from refiner.processors.step import RefinerStep, VectorizedOp, VectorizedSegmentStep
from refiner.runtime.execution.row_queue import RowQueue
from refiner.runtime.execution.row_steps import execute_row_steps
from refiner.runtime.execution.vectorized import (
    TabularBlock,
    apply_vectorized_op,
    iter_record_batch_rows,
    iter_table_rows,
    rows_to_table,
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
Block = list[Row] | TabularBlock
StreamItem = Row | Block


def compile_segments(steps: Sequence[RefinerStep]) -> tuple[Segment, ...]:
    # Vectorized segments are explicit fused nodes; everything else is executed
    # through the row-step engine until the next vectorized boundary.
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
) -> Iterator[Block]:
    current: Iterable[Block] = _normalize_blocks(
        stream,
        block_rows=vectorized_chunk_rows,
    )
    for idx, segment in enumerate(segments):
        next_segment = segments[idx + 1] if idx + 1 < len(segments) else None
        if isinstance(segment, VectorSegment):
            current = _execute_vector_segment(
                current,
                segment.ops,
                vectorized_chunk_rows=vectorized_chunk_rows,
            )
        else:
            current = _execute_row_segment(
                current,
                segment.steps,
                output_block_rows=vectorized_chunk_rows,
                output_tabular=isinstance(next_segment, VectorSegment),
            )
    yield from current


def iter_rows(stream: Iterable[StreamItem]) -> Iterator[Row]:
    for item in stream:
        if isinstance(item, Row):
            yield item
            continue
        if isinstance(item, list):
            yield from item
            continue
        if isinstance(item, pa.RecordBatch):
            yield from iter_record_batch_rows(item)
            continue
        if isinstance(item, pa.Table):
            yield from iter_table_rows(item)
            continue
        raise TypeError(f"Unsupported stream item: {type(item)!r}")


def block_num_rows(item: StreamItem) -> int:
    if isinstance(item, Row):
        return 1
    if isinstance(item, list):
        return len(item)
    if isinstance(item, pa.RecordBatch):
        return int(item.num_rows)
    if isinstance(item, pa.Table):
        return int(item.num_rows)
    raise TypeError(f"Unsupported stream item: {type(item)!r}")


def _normalize_blocks(
    stream: Iterable[StreamItem], *, block_rows: int
) -> Iterator[Block]:
    pending_rows: list[Row] = []

    def _flush_pending_rows() -> Iterator[list[Row]]:
        nonlocal pending_rows
        if not pending_rows:
            return
        batch = pending_rows
        pending_rows = []
        yield batch

    for item in stream:
        if isinstance(item, Row):
            pending_rows.append(item)
            if len(pending_rows) >= block_rows:
                yield from _flush_pending_rows()
            continue

        if pending_rows:
            yield from _flush_pending_rows()

        if isinstance(item, list):
            if item:
                yield item
            continue

        if isinstance(item, pa.RecordBatch):
            if item.num_rows > 0:
                yield item
            continue

        if isinstance(item, pa.Table):
            if item.num_rows > 0:
                yield item
            continue

        raise TypeError(f"Unsupported stream item: {type(item)!r}")

    if pending_rows:
        yield from _flush_pending_rows()


def _execute_row_segment(
    stream: Iterable[Block],
    steps: Sequence[RefinerStep],
    *,
    output_block_rows: int,
    output_tabular: bool,
) -> Iterator[Block]:
    # Row/UDF execution consumes row views and emits row blocks for downstream
    # vectorized segments (or final row iteration).
    rows = iter_rows(stream)
    step_out = execute_row_steps(rows, steps)
    if not output_tabular:
        yield from _chunk_output_rows(step_out, output_block_rows)
        return
    for batch in _chunk_output_rows(step_out, output_block_rows):
        table = rows_to_table(batch)
        if table.num_rows > 0:
            yield table


def _execute_vector_segment(
    stream: Iterable[Block],
    ops: Sequence[VectorizedOp],
    *,
    vectorized_chunk_rows: int,
) -> Iterator[TabularBlock]:
    pending_rows = RowQueue()

    def _run_block(block: TabularBlock) -> TabularBlock:
        out = block
        for op in ops:
            out = apply_vectorized_op(out, op)
        return out

    def _flush_rows() -> Iterator[TabularBlock]:
        if len(pending_rows) == 0:
            return
        batch = pending_rows.take_all()
        out = _run_block(rows_to_table(batch))
        if out.num_rows > 0:
            yield out

    for item in stream:
        if isinstance(item, list):
            pending_rows.extend(item)
            while len(pending_rows) >= vectorized_chunk_rows:
                chunk = pending_rows.take(vectorized_chunk_rows)
                out = _run_block(rows_to_table(chunk))
                if out.num_rows > 0:
                    yield out
            continue

        yield from _flush_rows()
        out = _run_block(item)
        if out.num_rows > 0:
            yield out

    yield from _flush_rows()


def _chunk_output_rows(rows: Iterable[Row], block_rows: int) -> Iterator[list[Row]]:
    pending: list[Row] = []
    for row in rows:
        pending.append(row)
        if len(pending) >= block_rows:
            out = pending
            pending = []
            yield out
    if pending:
        yield pending


__all__ = [
    "Block",
    "RowSegment",
    "VectorSegment",
    "block_num_rows",
    "compile_segments",
    "execute_segments",
    "iter_rows",
]
