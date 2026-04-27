from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from refiner.pipeline.data.block import Block, StreamItem
from refiner.pipeline.data.datatype import schema_with_dtypes
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.steps import (
    CastStep,
    DropStep,
    FnAsyncRowStep,
    FnBatchStep,
    FnFlatMapStep,
    FnRowStep,
    FnTableStep,
    RenameStep,
    RefinerStep,
    SelectStep,
    VectorizedOp,
    VectorizedSegmentStep,
    WithColumnsStep,
)
from refiner.execution.buffer import RowBuffer
from refiner.execution.operators.row import ShardDeltaFn, execute_row_steps
from refiner.execution.operators.vectorized import (
    apply_vectorized_ops,
)
from refiner.pipeline.data.row import Row

_DEFAULT_VECTORIZED_CHUNK_ROWS = 2048


@dataclass(frozen=True, slots=True)
class RowSegment:
    steps: tuple[RefinerStep, ...]


@dataclass(frozen=True, slots=True)
class VectorSegment:
    ops: tuple[VectorizedOp, ...]


Segment = RowSegment | VectorSegment


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
    max_vectorized_block_bytes: int | None = None,
    on_shard_delta: ShardDeltaFn | None = None,
    input_schema: pa.Schema | None = None,
    final_output_tabular: bool = False,
) -> Iterator[Block]:
    """Execute segments, optionally converting final row blocks to `Tabular` blocks."""
    current: Iterable[Block] = _normalize_blocks(
        stream,
        block_rows=vectorized_chunk_rows,
    )
    current_schema = input_schema
    for idx, segment in enumerate(segments):
        next_segment = segments[idx + 1] if idx + 1 < len(segments) else None
        if isinstance(segment, VectorSegment):
            current = _execute_vector_segment(
                current,
                segment.ops,
                vectorized_chunk_rows=vectorized_chunk_rows,
                max_vectorized_block_bytes=max_vectorized_block_bytes,
                on_shard_delta=on_shard_delta,
                input_schema=current_schema,
            )
            current_schema = _vector_segment_schema(current_schema, segment.ops)
        else:
            output_schema = _row_segment_schema(current_schema, segment.steps)
            current = _execute_row_segment(
                current,
                segment.steps,
                output_block_rows=vectorized_chunk_rows,
                output_tabular=isinstance(next_segment, VectorSegment)
                and max_vectorized_block_bytes is None,
                on_shard_delta=on_shard_delta,
                output_schema=output_schema,
            )
            current_schema = output_schema
    if final_output_tabular:
        yield from _tabularize_blocks(current, schema=current_schema)
        return
    yield from current


def iter_rows(stream: Iterable[StreamItem]) -> Iterator[Row]:
    for item in stream:
        if isinstance(item, Row):
            yield item
            continue
        if isinstance(item, list):
            yield from item
            continue
        if isinstance(item, Tabular):
            yield from item.to_rows()
            continue
        raise TypeError(f"Unsupported stream item: {type(item)!r}")


def block_num_rows(item: StreamItem) -> int:
    if isinstance(item, Row):
        return 1
    if isinstance(item, list):
        return len(item)
    if isinstance(item, Tabular):
        return item.num_rows
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

        if isinstance(item, Tabular):
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
    on_shard_delta: ShardDeltaFn | None,
    output_schema: pa.Schema | None,
) -> Iterator[Block]:
    # Row/UDF execution consumes row views and emits row blocks for downstream
    # vectorized segments (or final row iteration).
    rows = iter_rows(stream)
    step_out = execute_row_steps(rows, steps, on_shard_delta=on_shard_delta)
    if not output_tabular:
        yield from _chunk_output_rows(step_out, output_block_rows)
        return
    for batch in _chunk_output_rows(step_out, output_block_rows):
        block = (
            Tabular.from_rows(batch, schema=output_schema)
            if not batch
            else batch[0].tabular_type.from_rows(batch, schema=output_schema)
        )
        if block.table.num_rows > 0:
            yield block


def _row_segment_schema(
    input_schema: pa.Schema | None,
    steps: Sequence[RefinerStep],
) -> pa.Schema | None:
    schema = input_schema
    for step in steps:
        dtypes = (
            step.dtypes
            if isinstance(step, (FnRowStep, FnAsyncRowStep, FnBatchStep, FnFlatMapStep))
            else None
        )
        if not dtypes:
            continue
        schema = schema_with_dtypes(schema, dtypes)
    return schema


def _vector_segment_schema(
    input_schema: pa.Schema | None,
    ops: Sequence[VectorizedOp],
) -> pa.Schema | None:
    schema = input_schema
    for op in ops:
        if isinstance(op, FnTableStep):
            schema = None
            continue
        if isinstance(op, CastStep):
            schema = schema_with_dtypes(
                schema,
                op.dtypes,
                preserve_metadata=False,
            )
            continue
        if schema is None:
            continue
        if isinstance(op, SelectStep):
            fields = [
                schema.field(idx)
                for name in op.columns
                if (idx := schema.get_field_index(name)) >= 0
            ]
            schema = pa.schema(
                fields,
                metadata=schema.metadata,
            )
            continue
        if isinstance(op, DropStep):
            drop = set(op.columns)
            schema = pa.schema(
                [field for field in schema if field.name not in drop],
                metadata=schema.metadata,
            )
            continue
        if isinstance(op, RenameStep):
            schema = pa.schema(
                [
                    field.with_name(op.mapping.get(field.name, field.name))
                    for field in schema
                ],
                metadata=schema.metadata,
            )
            continue
        if isinstance(op, WithColumnsStep):
            assigned = set(op.assignments)
            schema = pa.schema(
                [field for field in schema if field.name not in assigned],
                metadata=schema.metadata,
            )
            continue
    return schema


def _execute_vector_segment(
    stream: Iterable[Block],
    ops: Sequence[VectorizedOp],
    *,
    vectorized_chunk_rows: int,
    max_vectorized_block_bytes: int | None,
    on_shard_delta: ShardDeltaFn | None,
    input_schema: pa.Schema | None,
) -> Iterator[Tabular]:
    pending_rows = RowBuffer()
    current_chunk_rows = max(1, int(vectorized_chunk_rows))
    estimated_row_bytes: float | None = None

    def _run_block(block: Tabular) -> Tabular:
        return block.with_table(
            apply_vectorized_ops(block.table, ops, on_shard_delta=on_shard_delta)
        )

    def _chunk_rows_for_budget() -> int:
        if (
            max_vectorized_block_bytes is None
            or estimated_row_bytes is None
            or estimated_row_bytes <= 0
        ):
            return current_chunk_rows
        budget_rows = int(max_vectorized_block_bytes / estimated_row_bytes)
        return max(1, min(current_chunk_rows, budget_rows))

    def _run_pending_chunk(target_rows: int) -> Iterator[Tabular]:
        nonlocal current_chunk_rows, estimated_row_bytes
        rows_for_try = max(1, target_rows)
        while True:
            batch = pending_rows.peek(rows_for_try)
            try:
                block = (
                    Tabular.from_rows(batch, schema=input_schema)
                    if not batch
                    else batch[0].tabular_type.from_rows(batch, schema=input_schema)
                )
            except pa.ArrowMemoryError:
                if rows_for_try <= 1:
                    raise
                rows_for_try = max(1, rows_for_try // 2)
                current_chunk_rows = min(current_chunk_rows, rows_for_try)
                continue
            table = block.table

            if table.num_rows > 0:
                estimated_row_bytes = table.nbytes / int(table.num_rows)

            if (
                max_vectorized_block_bytes is not None
                and table.num_rows > 1
                and table.nbytes > max_vectorized_block_bytes
            ):
                scaled_rows = int(
                    rows_for_try
                    * (max_vectorized_block_bytes / max(1, int(table.nbytes)))
                )
                rows_for_try = min(rows_for_try - 1, max(1, scaled_rows))
                current_chunk_rows = min(current_chunk_rows, rows_for_try)
                continue

            try:
                out = _run_block(block)
            except pa.ArrowMemoryError:
                if rows_for_try <= 1:
                    raise
                rows_for_try = max(1, rows_for_try // 2)
                current_chunk_rows = min(current_chunk_rows, rows_for_try)
                continue

            pending_rows.discard(rows_for_try)
            if out.table.num_rows > 0:
                yield out
            return

    def _drain_rows(*, force: bool) -> Iterator[Tabular]:
        while len(pending_rows) > 0:
            desired_rows = _chunk_rows_for_budget()
            if not force and len(pending_rows) < desired_rows:
                return
            yield from _run_pending_chunk(min(len(pending_rows), desired_rows))

    def _yield_tabular_chunks(block: Tabular) -> Iterator[Tabular]:
        nonlocal current_chunk_rows, estimated_row_bytes
        queue: deque[Tabular] = deque([block])
        while queue:
            chunk = queue.popleft()
            chunk_rows = int(chunk.table.num_rows)
            if chunk_rows <= 0:
                continue

            if (
                max_vectorized_block_bytes is not None
                and chunk_rows > 1
                and chunk.table.nbytes > max_vectorized_block_bytes
            ):
                scaled_rows = int(
                    chunk_rows
                    * (max_vectorized_block_bytes / max(1, int(chunk.table.nbytes)))
                )
                split_rows = min(chunk_rows - 1, max(1, scaled_rows))
                for start in range(0, chunk_rows, split_rows):
                    queue.append(chunk.with_table(chunk.table.slice(start, split_rows)))
                current_chunk_rows = min(current_chunk_rows, split_rows)
                continue

            try:
                out = _run_block(chunk)
            except pa.ArrowMemoryError:
                if chunk_rows <= 1:
                    raise
                split_rows = max(1, chunk_rows // 2)
                queue.appendleft(
                    chunk.with_table(
                        chunk.table.slice(split_rows, chunk_rows - split_rows)
                    )
                )
                queue.appendleft(chunk.with_table(chunk.table.slice(0, split_rows)))
                current_chunk_rows = min(current_chunk_rows, split_rows)
                continue

            estimated_row_bytes = chunk.table.nbytes / chunk_rows
            if out.table.num_rows > 0:
                yield out

    for item in stream:
        if not isinstance(item, Tabular):
            row_block = cast(list[Row], item)
            pending_rows.extend(row_block)
            yield from _drain_rows(force=False)
            continue

        yield from _drain_rows(force=True)
        yield from _yield_tabular_chunks(item)

    yield from _drain_rows(force=True)


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


def _tabularize_blocks(
    blocks: Iterable[Block],
    *,
    schema: pa.Schema | None,
) -> Iterator[Tabular]:
    for block in blocks:
        if isinstance(block, Tabular):
            yield block
            continue
        table = (
            Tabular.from_rows(block, schema=schema)
            if not block
            else block[0].tabular_type.from_rows(block, schema=schema)
        )
        if table.num_rows > 0:
            yield table


__all__ = [
    "Block",
    "RowSegment",
    "VectorSegment",
    "block_num_rows",
    "compile_segments",
    "execute_segments",
    "iter_rows",
]
