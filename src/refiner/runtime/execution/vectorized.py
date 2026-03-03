from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa
import pyarrow.compute as pc

from refiner.expressions import eval_expr_arrow
from refiner.processors.step import (
    CastStep,
    DropStep,
    FilterExprStep,
    RenameStep,
    SelectStep,
    VectorizedOp,
    WithColumnsStep,
)
from refiner.sources.row import ArrowRowView, Row


def rows_to_table(rows: Iterable[Row]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in rows])


def table_to_rows(table: pa.Table) -> list[Row]:
    names = tuple(str(name) for name in table.column_names)
    columns = tuple(table.column(name) for name in names)
    index_by_name = {name: i for i, name in enumerate(names)}
    out: list[Row] = []
    for idx in range(table.num_rows):
        out.append(
            ArrowRowView(
                names=names,
                columns=columns,
                index_by_name=index_by_name,
                row_idx=idx,
            )
        )
    return out


def record_batch_to_rows(batch: pa.RecordBatch) -> list[Row]:
    # Same view model as table_to_rows, but avoids temporary Table allocation.
    names = tuple(str(name) for name in batch.schema.names)
    columns = tuple(batch.column(i) for i in range(batch.num_columns))
    index_by_name = {name: i for i, name in enumerate(names)}
    out: list[Row] = []
    for idx in range(batch.num_rows):
        out.append(
            ArrowRowView(
                names=names,
                columns=columns,
                index_by_name=index_by_name,
                row_idx=idx,
            )
        )
    return out


def apply_vectorized_op(table: pa.Table, op: VectorizedOp) -> pa.Table:
    if isinstance(op, SelectStep):
        return table.select(list(op.columns))

    if isinstance(op, DropStep):
        return table.drop(list(op.columns))

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in table.column_names]
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


def _broadcast_scalar(
    values: pa.Array | pa.ChunkedArray | pa.Scalar, num_rows: int
) -> pa.Array | pa.ChunkedArray:
    if not isinstance(values, pa.Scalar):
        return values
    return pa.array([values.as_py()] * num_rows, type=values.type)


def chunk_rows(rows: list[Row], chunk_size: int) -> list[list[Row]]:
    if chunk_size <= 0:
        return [rows]
    out: list[list[Row]] = []
    for i in range(0, len(rows), chunk_size):
        out.append(rows[i : i + chunk_size])
    return out


__all__ = [
    "rows_to_table",
    "table_to_rows",
    "record_batch_to_rows",
    "apply_vectorized_op",
    "chunk_rows",
]
