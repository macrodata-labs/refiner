from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa
import pyarrow.compute as pc

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
from refiner.pipeline.data.row import Row


def rows_to_block(rows: Iterable[Row]) -> Tabular:
    materialized = list(rows)
    if not materialized:
        return Tabular.from_rows(materialized)
    return materialized[0].tabular_type.from_rows(materialized)


def apply_vectorized_op(block: Tabular, op: VectorizedOp) -> Tabular:
    table = block.table
    if isinstance(op, SelectStep):
        return block.with_table(table.select(list(op.columns)))

    if isinstance(op, DropStep):
        return block.with_table(table.drop_columns(list(op.columns)))

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in table.schema.names]
        return block.with_table(table.rename_columns(names))

    if isinstance(op, CastStep):
        out = table
        for col_name, dtype in op.dtypes.items():
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                raise KeyError(f"Unknown column for cast: {col_name}")
            casted = pc.cast(out.column(col_name), target_type=pa.type_for_alias(dtype))
            out = out.set_column(idx, col_name, casted)
        return block.with_table(out)

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
        return block.with_table(out)

    if isinstance(op, FilterExprStep):
        mask = eval_expr_arrow(op.predicate, table)
        if isinstance(mask, pa.Scalar):
            return block if bool(mask.as_py()) else block.with_table(table.slice(0, 0))
        return block.with_table(table.filter(mask))

    raise TypeError(f"Unsupported vectorized op: {type(op)!r}")


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
    "rows_to_block",
    "apply_vectorized_op",
]
