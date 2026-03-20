from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from refiner.execution.tracking.shards import (
    ShardDeltaFn,
    count_table_by_shard,
    counts_delta,
)
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
from refiner.worker.metrics.api import log_throughput


def apply_vectorized_op(
    table: pa.Table,
    op: VectorizedOp,
    *,
    shard_counts: dict[str, int] | None = None,
) -> tuple[pa.Table, dict[str, int] | None]:
    if shard_counts is None:
        shard_counts = count_table_by_shard(table)

    if isinstance(op, SelectStep):
        return table.select(list(op.columns)), None

    if isinstance(op, DropStep):
        return table.drop_columns(list(op.columns)), None

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in table.schema.names]
        return table.rename_columns(names), None

    if isinstance(op, CastStep):
        out = table
        for col_name, dtype in op.dtypes.items():
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                raise KeyError(f"Unknown column for cast: {col_name}")
            casted = pc.cast(out.column(col_name), target_type=pa.type_for_alias(dtype))
            out = out.set_column(idx, col_name, casted)
        return out, None

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
                out = out.set_column(idx, col_name, values)
        return out, None

    if isinstance(op, FilterExprStep):
        mask = eval_expr_arrow(op.predicate, table)
        next_table = (
            table
            if isinstance(mask, pa.Scalar) and bool(mask.as_py())
            else (
                table.slice(0, 0) if isinstance(mask, pa.Scalar) else table.filter(mask)
            )
        )
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
        return next_table, next_shard_counts

    if isinstance(op, FnTableStep):
        next_table = op.fn(table)
        if not isinstance(next_table, pa.Table):
            raise TypeError(
                f"map_table() must return pa.Table, got {type(next_table)!r}"
            )
        next_shard_counts = count_table_by_shard(next_table)
        return next_table, next_shard_counts

    raise TypeError(f"Unsupported vectorized op: {type(op)!r}")


def apply_vectorized_ops(
    table: pa.Table,
    ops: Sequence[VectorizedOp],
    *,
    on_shard_delta: ShardDeltaFn | None = None,
) -> pa.Table:
    initial_shard_counts = count_table_by_shard(table)
    shard_counts = initial_shard_counts
    for op in ops:
        for shard_id, count in shard_counts.items():
            log_throughput(
                "rows_processed",
                count,
                shard_id=shard_id,
                unit="rows",
                step_index=op.index,
            )
        table, next_shard_counts = apply_vectorized_op(
            table,
            op,
            shard_counts=shard_counts,
        )
        if next_shard_counts is not None:
            shard_counts = next_shard_counts
    if on_shard_delta is not None:
        delta = counts_delta(produced=shard_counts, consumed=initial_shard_counts)
        if delta:
            on_shard_delta(delta)
    return table


__all__ = [
    "apply_vectorized_op",
    "apply_vectorized_ops",
]
