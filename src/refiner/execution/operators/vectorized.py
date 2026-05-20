from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pyarrow as pa

from refiner.execution.tracking.shards import (
    ShardDeltaFn,
    count_table_by_shard,
    counts_delta,
)
from refiner.pipeline.data.datatype import apply_dtypes_to_table
from refiner.pipeline.data.tabular import repeat_scalar
from refiner.pipeline.expressions import eval_expr_arrow
from refiner.pipeline.steps import (
    CastStep,
    DropStep,
    FilterExprStep,
    FnTableStep,
    RenameStep,
    SelectStep,
    VectorizedOp,
    WithColumnsStep,
)
from refiner.worker.context import set_active_step_index
from refiner.worker.metrics.api import log_throughput

_ROW_INDEX_COLUMN = "__refiner_row_index"
RowIndices = tuple[int, ...] | None


def _identity_row_indices(row_indices: RowIndices, num_rows: int) -> RowIndices:
    return None if row_indices == tuple(range(num_rows)) else row_indices


def apply_vectorized_op(
    table: pa.Table,
    op: VectorizedOp,
    *,
    shard_counts: dict[str, int] | None = None,
    row_indices: RowIndices = None,
    return_row_indices: bool = False,
) -> tuple[pa.Table, dict[str, int] | None, RowIndices]:
    if shard_counts is None:
        shard_counts = count_table_by_shard(table)

    if isinstance(op, SelectStep):
        return table.select(list(op.columns)), None, row_indices

    if isinstance(op, DropStep):
        return table.drop_columns(list(op.columns)), None, row_indices

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in table.schema.names]
        return table.rename_columns(names), None, row_indices

    if isinstance(op, CastStep):
        return (
            apply_dtypes_to_table(
                table,
                op.dtypes,
                preserve_metadata=False,
            ),
            None,
            row_indices,
        )

    if isinstance(op, WithColumnsStep):
        out = table
        for col_name, expr in op.assignments.items():
            values = eval_expr_arrow(expr, out)
            if isinstance(values, pa.Scalar):
                values = repeat_scalar(values, out.num_rows)
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                out = out.append_column(col_name, values)
            else:
                out = out.set_column(idx, pa.field(col_name, values.type), values)
        return out, None, row_indices

    if isinstance(op, FilterExprStep):
        mask = eval_expr_arrow(op.predicate, table)
        if isinstance(mask, pa.Scalar):
            keep_all = bool(mask.as_py())
            next_table = table if keep_all else table.slice(0, 0)
            next_row_indices = row_indices if keep_all else ()
        else:
            if isinstance(mask, pa.ChunkedArray):
                mask = mask.combine_chunks()
            next_table = table.filter(mask)
            if return_row_indices:
                kept = tuple(
                    int(idx)
                    for idx in np.flatnonzero(
                        mask.fill_null(False).to_numpy(zero_copy_only=False)
                    )
                )
                next_row_indices = (
                    _identity_row_indices(kept, table.num_rows)
                    if row_indices is None
                    else tuple(row_indices[idx] for idx in kept)
                )
            else:
                next_row_indices = None
        if not return_row_indices:
            next_row_indices = None
        next_shard_counts = count_table_by_shard(next_table)
        for shard_id in set(shard_counts) | set(next_shard_counts):
            previous = int(shard_counts.get(shard_id, 0))
            current = int(next_shard_counts.get(shard_id, 0))
            if current > 0:
                log_throughput(
                    "rows_kept",
                    current,
                    shard_id=shard_id,
                    unit="rows",
                    step_index=op.index,
                )
            dropped = previous - current
            if dropped > 0:
                log_throughput(
                    "rows_dropped",
                    dropped,
                    shard_id=shard_id,
                    unit="rows",
                    step_index=op.index,
                )
        return next_table, next_shard_counts, next_row_indices

    if isinstance(op, FnTableStep):
        if return_row_indices:
            if _ROW_INDEX_COLUMN in table.column_names:
                raise ValueError(f"{_ROW_INDEX_COLUMN} is an internal column")
            lineage = range(table.num_rows) if row_indices is None else row_indices
            table = table.append_column(
                _ROW_INDEX_COLUMN,
                pa.array(lineage, type=pa.int64()),
            )
        with set_active_step_index(op.index):
            next_table = op.fn(table)
        if not isinstance(next_table, pa.Table):
            raise TypeError(
                f"map_table() must return pa.Table, got {type(next_table)!r}"
            )
        next_row_indices = None
        if return_row_indices:
            if _ROW_INDEX_COLUMN not in next_table.column_names:
                raise ValueError(
                    f"map_table() must preserve {_ROW_INDEX_COLUMN} for this input"
                )
            lineage_column = next_table.column(_ROW_INDEX_COLUMN).combine_chunks()
            lineage = tuple(
                int(value) for value in lineage_column.to_numpy(zero_copy_only=False)
            )
            next_row_indices = (
                _identity_row_indices(lineage, table.num_rows)
                if row_indices is None
                else lineage
            )
            next_table = next_table.drop_columns([_ROW_INDEX_COLUMN])
        next_shard_counts = count_table_by_shard(next_table)
        return next_table, next_shard_counts, next_row_indices

    raise TypeError(f"Unsupported vectorized op: {type(op)!r}")


@overload
def apply_vectorized_ops(
    table: pa.Table,
    ops: Sequence[VectorizedOp],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
    return_row_indices: Literal[False] = False,
) -> pa.Table: ...


@overload
def apply_vectorized_ops(
    table: pa.Table,
    ops: Sequence[VectorizedOp],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
    return_row_indices: Literal[True],
) -> tuple[pa.Table, tuple[int, ...] | None]: ...


def apply_vectorized_ops(
    table: pa.Table,
    ops: Sequence[VectorizedOp],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
    return_row_indices: bool = False,
) -> pa.Table | tuple[pa.Table, tuple[int, ...] | None]:
    initial_shard_counts = count_table_by_shard(table)
    shard_counts = initial_shard_counts
    # Some Tabular subclasses keep per-row side data outside the Arrow table.
    # Row indices let them realign that side data after vectorized filters/reorders.
    row_indices: RowIndices = None
    initial_rows = table.num_rows
    for op in ops:
        for shard_id, count in shard_counts.items():
            log_throughput(
                "rows_processed",
                count,
                shard_id=shard_id,
                unit="rows",
                step_index=op.index,
            )
        table, next_shard_counts, row_indices = apply_vectorized_op(
            table,
            op,
            shard_counts=shard_counts,
            row_indices=row_indices,
            return_row_indices=return_row_indices,
        )
        if next_shard_counts is not None:
            shard_counts = next_shard_counts
    if on_shard_delta is not None:
        delta = counts_delta(produced=shard_counts, consumed=initial_shard_counts)
        if delta:
            on_shard_delta(delta)
    if not return_row_indices:
        return table
    if table.num_rows == initial_rows:
        row_indices = _identity_row_indices(row_indices, initial_rows)
    return table, row_indices


__all__ = [
    "apply_vectorized_op",
    "apply_vectorized_ops",
]
