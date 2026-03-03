from __future__ import annotations

from collections.abc import Iterable, Iterator

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

TabularBlock = pa.Table | pa.RecordBatch


def rows_to_table(rows: Iterable[Row]) -> pa.Table:
    if isinstance(rows, list):
        return pa.Table.from_pylist(rows)
    return pa.Table.from_pylist(list(rows))


def iter_table_rows(table: pa.Table) -> Iterator[Row]:
    names = tuple(str(name) for name in table.column_names)
    columns = tuple(table.column(name) for name in names)
    index_by_name = {name: i for i, name in enumerate(names)}
    for idx in range(table.num_rows):
        yield ArrowRowView(
            names=names,
            columns=columns,
            index_by_name=index_by_name,
            row_idx=idx,
        )


def iter_record_batch_rows(batch: pa.RecordBatch) -> Iterator[Row]:
    # Same row-view model as tables, without temporary Table allocation.
    names = tuple(str(name) for name in batch.schema.names)
    columns = tuple(batch.column(i) for i in range(batch.num_columns))
    index_by_name = {name: i for i, name in enumerate(names)}
    for idx in range(batch.num_rows):
        yield ArrowRowView(
            names=names,
            columns=columns,
            index_by_name=index_by_name,
            row_idx=idx,
        )


def apply_vectorized_op(block: TabularBlock, op: VectorizedOp) -> TabularBlock:
    if isinstance(op, SelectStep):
        return block.select(list(op.columns))

    if isinstance(op, DropStep):
        return block.drop_columns(list(op.columns))

    if isinstance(op, RenameStep):
        names = [op.mapping.get(name, name) for name in block.schema.names]
        return block.rename_columns(names)

    if isinstance(op, CastStep):
        out = block
        for col_name, dtype in op.dtypes.items():
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                raise KeyError(f"Unknown column for cast: {col_name}")
            casted = pc.cast(out.column(col_name), target_type=pa.type_for_alias(dtype))
            out = out.set_column(idx, col_name, casted)
        return out

    if isinstance(op, WithColumnsStep):
        out = block
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
        mask = eval_expr_arrow(op.predicate, block)
        if isinstance(mask, pa.Scalar):
            return block if bool(mask.as_py()) else block.slice(0, 0)
        return block.filter(mask)

    raise TypeError(f"Unsupported vectorized op: {type(op)!r}")


def _broadcast_scalar(
    values: pa.Array | pa.ChunkedArray | pa.Scalar, num_rows: int
) -> pa.Array | pa.ChunkedArray:
    if not isinstance(values, pa.Scalar):
        return values
    if num_rows <= 0:
        return pa.array([], type=values.type)
    # Broadcast without materializing a Python list of repeated values.
    all_rows = pc.call_function("is_null", [pa.nulls(num_rows)])
    return pc.call_function("if_else", [all_rows, values, values])


__all__ = [
    "TabularBlock",
    "rows_to_table",
    "iter_table_rows",
    "iter_record_batch_rows",
    "apply_vectorized_op",
]
