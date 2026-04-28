from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import count
from typing import Any, cast

import pyarrow as pa
import numpy as np

from refiner.pipeline.expressions import Expr, eval_expr_arrow
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.row import ArrowRowView, Row, _OverlayRow

_NEXT_TABULAR_ID = count()


class Tabular:
    unit: pa.Table

    def __init__(self, unit: pa.Table) -> None:
        self.unit = unit
        self.tabular_id = next(_NEXT_TABULAR_ID)
        self.names = tuple(str(name) for name in unit.column_names)
        self.columns = tuple(unit.column(name) for name in self.names)
        self.index_by_name = {name: i for i, name in enumerate(self.names)}
        self.shard_idx = self.index_by_name.get(SHARD_ID_COLUMN)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Row],
        *,
        schema: pa.Schema | None = None,
    ) -> "Tabular":
        if not rows:
            return cls(pa.table({}))
        if all(_is_arrow_backed(row) for row in rows):
            # Fast path for Arrow-backed rows: sort by backing tabular source and row index so
            # we can rebuild with slice/take instead of materializing every cell in Python.
            tables = _arrow_tables_from_rows(
                _sorted_arrow_rows(rows),
                schema=schema,
            )
            if len(tables) == 1:
                return cls(tables[0])
            return cls(_concat_tables(tables))
        # Generic row fallback. A union-of-names pass plus row.get(...) was as fast as the
        # earlier DictRow-specific special case and simpler to keep correct for mixed rows.
        return cls(_table_from_rows(rows, schema=schema))

    @classmethod
    def from_batch(cls, batch: pa.RecordBatch) -> "Tabular":
        return cls(pa.Table.from_batches([batch]))

    @property
    def table(self) -> pa.Table:
        return self.unit

    @property
    def num_rows(self) -> int:
        return int(self.unit.num_rows)

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)

    @property
    def schema(self) -> pa.Schema:
        return self.unit.schema

    def column(self, name: str) -> pa.Array | pa.ChunkedArray:
        return self.unit.column(name)

    def __iter__(self) -> Iterator[Row]:
        for row_idx in range(self.num_rows):
            shard_id = None
            if self.shard_idx is not None:
                shard = self.columns[self.shard_idx][row_idx].as_py()
                shard_id = shard if isinstance(shard, str) else None
            yield ArrowRowView(
                tabular=self,
                row_idx=row_idx,
                shard_id=shard_id,
            )

    def to_rows(self) -> list[Row]:
        return list(self)

    def with_table(self, table: pa.Table) -> "Tabular":
        return Tabular(table)


def set_or_append_column(
    table: pa.Table,
    name: str,
    column: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    if name in table.column_names:
        idx = table.column_names.index(name)
        current = table.schema.field(idx)
        metadata = current.metadata if current.type == column.type else None
        field = pa.field(
            name,
            column.type,
            nullable=current.nullable or column.null_count > 0,
            metadata=metadata,
        )
        return table.set_column(idx, field, column)
    return table.append_column(name, column)


def repeat_scalar(value: pa.Scalar, num_rows: int) -> pa.Array | pa.ChunkedArray:
    if num_rows <= 0:
        return pa.array([], type=value.type)
    return pa.repeat(value, num_rows)


def filter_table(table: pa.Table, predicate: Expr) -> pa.Table:
    mask = eval_expr_arrow(predicate, table)
    if isinstance(mask, pa.Scalar):
        return table if bool(mask.as_py()) else table.slice(0, 0)
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    return table.filter(mask)


# everything below is for fast from_rows
def _table_from_rows(
    rows: Sequence[Row],
    *,
    schema: pa.Schema | None = None,
) -> pa.Table:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name == SHARD_ID_COLUMN or name in seen:
                continue
            seen.add(name)
            names.append(name)
    if schema is None:
        columns = {
            name: _array_from_values([row.get(name) for row in rows], None, None)[0]
            for name in names
        }
        if any(row.shard_id is not None for row in rows):
            columns[SHARD_ID_COLUMN] = [row.shard_id for row in rows]
        return pa.table(columns)

    arrays: dict[str, pa.Array] = {}
    metadata_by_name: dict[str, dict[bytes, bytes]] = {}
    for name in names:
        schema_field = _schema_field(schema, name)
        column, field_metadata = _array_from_values(
            [row.get(name) for row in rows],
            schema_field.type if schema_field is not None else None,
            schema_field.metadata if schema_field is not None else None,
        )
        arrays[name] = column
        if field_metadata:
            metadata_by_name[name] = field_metadata
    if any(row.shard_id is not None for row in rows):
        arrays[SHARD_ID_COLUMN] = pa.array(
            [row.shard_id for row in rows],
            type=pa.string(),
        )
    if not metadata_by_name:
        return pa.table(arrays)
    fields = [
        pa.field(name, column.type, metadata=metadata_by_name.get(name))
        for name, column in arrays.items()
    ]
    return pa.table(arrays, schema=pa.schema(fields))


def _concat_tables(tables: Sequence[pa.Table]) -> pa.Table:
    try:
        return pa.concat_tables(tables)
    except pa.ArrowInvalid:
        return pa.concat_tables(tables, promote_options="default")


def _sorted_arrow_rows(rows: Sequence[Row]) -> Sequence[Row]:
    return sorted(
        rows,
        key=lambda row: (
            _base_arrow_row(row).tabular.tabular_id,
            row.shard_id or "",
            _base_arrow_row(row).row_idx,
        ),
    )


def _arrow_tables_from_rows(
    rows: Sequence[Row],
    *,
    schema: pa.Schema | None = None,
) -> list[pa.Table]:
    tables: list[pa.Table] = []
    group_start = 0
    while group_start < len(rows):
        first = rows[group_start]
        first_base = _base_arrow_row(first)
        group_tabular_id = first_base.tabular.tabular_id
        group_shard_id = first.shard_id
        group_end = group_start + 1
        while group_end < len(rows):
            candidate = rows[group_end]
            candidate_base = _base_arrow_row(candidate)
            if candidate_base.tabular.tabular_id != group_tabular_id:
                break
            if candidate.shard_id != group_shard_id:
                break
            group_end += 1
        tables.append(
            _arrow_table_from_group(
                rows[group_start:group_end],
                schema=schema,
            )
        )
        group_start = group_end
    return tables


def _arrow_table_from_group(
    rows: Sequence[Row],
    *,
    schema: pa.Schema | None = None,
) -> pa.Table:
    sample = _base_arrow_row(rows[0])
    base = sample.tabular.table
    row_indices = [_base_arrow_row(row).row_idx for row in rows]
    min_idx = row_indices[0]
    max_idx = row_indices[-1]
    span = max_idx - min_idx + 1
    density = len(row_indices) / span
    # Benchmarks over contiguous, interleaved, random, and patched sparsity patterns showed
    # that slice+take only wins once the kept rows are very dense inside the span.
    if density >= 0.85:
        sliced = base.slice(min_idx, span)
        relative_indices = [row_idx - min_idx for row_idx in row_indices]
        if _is_contiguous(row_indices):
            table = sliced
        else:
            table = sliced.take(pa.array(relative_indices, type=pa.int64()))
    else:
        table = base.take(pa.array(row_indices, type=pa.int64()))

    changed_columns = (
        _overlay_changes_by_column(rows)
        if any(isinstance(row, _OverlayRow) for row in rows)
        else {}
    )
    if schema is None and not changed_columns:
        return _with_shard_id(table, rows[0].shard_id, len(rows))

    for name, changes in changed_columns.items():
        if all(name not in row for row in rows):
            if name in table.column_names:
                table = table.drop([name])
            continue

        schema_field = _schema_field(schema, name)
        base_field = _schema_field(base.schema, name)
        field = schema_field if schema_field is not None else base_field
        value_type = field.type if field is not None else None
        if name in table.column_names:
            values = table[name].to_pylist()
        else:
            values = [None] * len(rows)
        for idx, value in changes:
            values[idx] = value
        column, field_metadata = _array_from_values(
            values,
            value_type,
            field.metadata if field is not None else None,
        )
        if schema_field is not None:
            field = pa.field(name, column.type, metadata=field_metadata)
            if name in table.column_names:
                table = table.set_column(table.column_names.index(name), field, column)
            else:
                table = table.append_column(field, column)
        elif name in table.column_names:
            table = table.set_column(
                table.column_names.index(name),
                pa.field(name, column.type, metadata=field_metadata),
                column,
            )
        else:
            table = set_or_append_column(table, name, column)

    table = _apply_schema_to_unchanged_columns(
        table,
        schema=schema,
        changed_names=set(changed_columns),
    )

    return _with_shard_id(table, rows[0].shard_id, len(rows))


def _with_shard_id(
    table: pa.Table,
    shard_id: str | None,
    num_rows: int,
) -> pa.Table:
    if shard_id is not None:
        shard_col = pa.array([shard_id] * num_rows, type=pa.string())
        table = set_or_append_column(table, SHARD_ID_COLUMN, shard_col)
    return table


def _overlay_changes_by_column(
    rows: Sequence[Row],
) -> dict[str, list[tuple[int, object]]]:
    # Sparse patch application is cheaper than rebuilding every column through row lookups.
    changes: dict[str, list[tuple[int, object]]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, _OverlayRow):
            continue
        for name in row.patch:
            changes.setdefault(name, []).append((idx, row.patch[name]))
        for name in row.deleted:
            changes.setdefault(name, []).append((idx, None))
    return changes


def _schema_field(schema: pa.Schema | None, name: str) -> pa.Field | None:
    if schema is None:
        return None
    idx = schema.get_field_index(name)
    return schema.field(idx) if idx >= 0 else None


def _apply_schema_to_unchanged_columns(
    table: pa.Table,
    *,
    schema: pa.Schema | None,
    changed_names: set[str],
) -> pa.Table:
    if schema is None:
        return table

    out = table
    for field in schema:
        name = field.name
        if name in changed_names:
            continue
        idx = out.schema.get_field_index(name)
        if idx < 0:
            continue
        column = out.column(name)
        try:
            if column.type != field.type:
                column = column.cast(field.type)
            replacement = pa.field(
                name,
                column.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            replacement = pa.field(name, column.type)
        if out.schema.field(idx).equals(replacement, check_metadata=True):
            continue
        out = out.set_column(idx, replacement, column)
    return out


def _array_from_values(
    values: Sequence[object],
    value_type: pa.DataType | None,
    metadata: dict[bytes, bytes] | None,
) -> tuple[pa.Array, dict[bytes, bytes] | None]:
    values = [_arrow_compatible_value(value) for value in values]
    if value_type is None:
        return pa.array(values), None
    try:
        return pa.array(values, type=value_type), dict(metadata) if metadata else None
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        inferred = pa.array(values)
        if metadata and b"asset_type" in metadata:
            return inferred, None
        try:
            return inferred.cast(value_type), dict(metadata) if metadata else None
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            return inferred, None


def _arrow_compatible_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return cast(Any, value).tolist()
    if isinstance(value, np.generic):
        return cast(Any, value).item()
    return value


def _is_arrow_backed(row: Row) -> bool:
    return isinstance(row, ArrowRowView) or (
        isinstance(row, _OverlayRow) and isinstance(row.base, ArrowRowView)
    )


def _base_arrow_row(row: Row) -> ArrowRowView:
    if isinstance(row, ArrowRowView):
        return row
    assert isinstance(row, _OverlayRow)
    assert isinstance(row.base, ArrowRowView)
    return row.base


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return True
    start = indices[0]
    return indices == list(range(start, start + len(indices)))


__all__ = ["Tabular"]
