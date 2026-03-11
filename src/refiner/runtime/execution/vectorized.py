from __future__ import annotations

from collections.abc import Iterable, Iterator

import pyarrow as pa
import pyarrow.compute as pc

from refiner.ledger.shard_tracking import SHARD_ID_COLUMN
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
from refiner.runtime.types import TabularBlock
from refiner.sources.row import ArrowRowView, Row


def rows_to_table(rows: Iterable[Row]) -> pa.Table:
    materialized = rows if isinstance(rows, list) else list(rows)
    table = pa.Table.from_pylist(materialized)
    if SHARD_ID_COLUMN in table.schema.names:
        return table
    if not materialized:
        return table
    shard_ids = [row.shard_id for row in materialized]
    if not any(shard is not None for shard in shard_ids):
        return table
    if any(shard is None for shard in shard_ids):
        raise ValueError(
            "rows_to_table requires shard_id for all rows when shard tracking is active"
        )
    return table.append_column(SHARD_ID_COLUMN, pa.array(shard_ids, type=pa.string()))


def iter_table_rows(table: pa.Table) -> Iterator[Row]:
    names = tuple(str(name) for name in table.column_names)
    columns = tuple(table.column(name) for name in names)
    index_by_name = {name: i for i, name in enumerate(names)}
    shard_col = (
        columns[index_by_name[SHARD_ID_COLUMN]]
        if SHARD_ID_COLUMN in index_by_name
        else None
    )
    for idx in range(table.num_rows):
        shard_id = None if shard_col is None else shard_col[idx].as_py()
        yield ArrowRowView(
            names=names,
            columns=columns,
            index_by_name=index_by_name,
            row_idx=idx,
            shard_id=shard_id if isinstance(shard_id, str) else None,
        )


def iter_record_batch_rows(batch: pa.RecordBatch) -> Iterator[Row]:
    # Same row-view model as tables, without temporary Table allocation.
    names = tuple(str(name) for name in batch.schema.names)
    columns = tuple(batch.column(i) for i in range(batch.num_columns))
    index_by_name = {name: i for i, name in enumerate(names)}
    shard_col = (
        columns[index_by_name[SHARD_ID_COLUMN]]
        if SHARD_ID_COLUMN in index_by_name
        else None
    )
    for idx in range(batch.num_rows):
        shard_id = None if shard_col is None else shard_col[idx].as_py()
        yield ArrowRowView(
            names=names,
            columns=columns,
            index_by_name=index_by_name,
            row_idx=idx,
            shard_id=shard_id if isinstance(shard_id, str) else None,
        )


def apply_vectorized_op(block: TabularBlock, op: VectorizedOp) -> TabularBlock:
    if isinstance(op, SelectStep):
        cols = list(op.columns)
        if SHARD_ID_COLUMN in block.schema.names and SHARD_ID_COLUMN not in cols:
            cols.append(SHARD_ID_COLUMN)
        return block.select(cols)

    if isinstance(op, DropStep):
        cols = [name for name in op.columns if name != SHARD_ID_COLUMN]
        return block.drop_columns(cols)

    if isinstance(op, RenameStep):
        names = [
            (SHARD_ID_COLUMN if name == SHARD_ID_COLUMN else op.mapping.get(name, name))
            for name in block.schema.names
        ]
        return block.rename_columns(names)

    if isinstance(op, CastStep):
        out = block
        for col_name, dtype in op.dtypes.items():
            if col_name == SHARD_ID_COLUMN:
                continue
            idx = out.schema.get_field_index(col_name)
            if idx < 0:
                raise KeyError(f"Unknown column for cast: {col_name}")
            casted = pc.cast(out.column(col_name), target_type=pa.type_for_alias(dtype))
            out = out.set_column(idx, col_name, casted)
        return out

    if isinstance(op, WithColumnsStep):
        out = block
        for col_name, expr in op.assignments.items():
            if col_name == SHARD_ID_COLUMN:
                continue
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
    return pa.repeat(values, num_rows)


__all__ = [
    "TabularBlock",
    "rows_to_table",
    "iter_table_rows",
    "iter_record_batch_rows",
    "apply_vectorized_op",
]
