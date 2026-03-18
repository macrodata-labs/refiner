from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.compute as pc

from refiner.execution.tracking.shards import count_tabular_by_shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.expressions import eval_expr_arrow
from refiner.pipeline.steps import (
    CastStep,
    DropStep,
    FilterExprStep,
    RenameStep,
    SelectStep,
    VectorizedOp,
    WithColumnsStep,
)
from refiner.worker.context import set_active_step_index
from refiner.worker.metrics.api import log_throughput


def apply_vectorized_op(table: pa.Table, op: VectorizedOp) -> pa.Table:
    if isinstance(op, SelectStep):
        return table.select(list(op.columns))

    if isinstance(op, DropStep):
        return table.drop_columns(list(op.columns))

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in table.schema.names]
        return table.rename_columns(names)

    if isinstance(op, CastStep):
        out = table
        for col_name, dtype in op.dtypes.items():
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                raise KeyError(f"Unknown column for cast: {col_name}")
            casted = pc.cast(out.column(col_name), target_type=pa.type_for_alias(dtype))
            out = out.set_column(idx, col_name, casted)
        return out

    if isinstance(op, WithColumnsStep):
        out = table
        for col_name, expr in op.assignments.items():
            values = eval_expr_arrow(expr, out)
            # Keep scalar expressions column-shaped for append/set_column.
            values = _broadcast_scalar(values, out.num_rows)
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                out = out.append_column(col_name, values)
            else:
                out = out.set_column(idx, col_name, values)
        return out

    if isinstance(op, FilterExprStep):
        mask = eval_expr_arrow(op.predicate, table)
        if isinstance(mask, pa.Scalar):
            return table if bool(mask.as_py()) else table.slice(0, 0)
        return table.filter(mask)

    raise TypeError(f"Unsupported vectorized op: {type(op)!r}")


def apply_vectorized_ops(block: Tabular, ops: Sequence[VectorizedOp]) -> Tabular:
    out = block
    for op in ops:
        with set_active_step_index(op.index) if hasattr(op, "index") else nullcontext():
            consumed_counts = count_tabular_by_shard(out)
            produced = out.with_table(apply_vectorized_op(out.table, op))
            produced_counts = count_tabular_by_shard(produced)
            for shard_id, count in produced_counts.items():
                log_throughput("rows_out", count, shard_id=shard_id, unit="rows")
            if isinstance(op, FilterExprStep):
                for shard_id in set(consumed_counts) | set(produced_counts):
                    dropped = int(consumed_counts.get(shard_id, 0)) - int(
                        produced_counts.get(shard_id, 0)
                    )
                    if dropped > 0:
                        log_throughput(
                            "rows_dropped",
                            dropped,
                            shard_id=shard_id,
                            unit="rows",
                        )
            out = produced
    return out


def _broadcast_scalar(
    values: pa.Array | pa.ChunkedArray | pa.Scalar, num_rows: int
) -> pa.Array | pa.ChunkedArray:
    if not isinstance(values, pa.Scalar):
        return values
    if num_rows <= 0:
        return pa.array([], type=values.type)
    return pa.repeat(values, num_rows)


__all__ = [
    "Tabular",
    "apply_vectorized_op",
    "apply_vectorized_ops",
]
