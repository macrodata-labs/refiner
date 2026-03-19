from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import count

import pyarrow as pa

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
    def from_rows(cls, rows: Sequence[Row]) -> "Tabular":
        if not rows:
            return cls(pa.table({}))
        if all(_is_arrow_backed(row) for row in rows):
            # Fast path for Arrow-backed rows: sort by backing tabular source and row index so
            # we can rebuild with slice/take instead of materializing every cell in Python.
            tables = _arrow_tables_from_rows(_sorted_arrow_rows(rows))
            if len(tables) == 1:
                return cls(tables[0])
            return cls(pa.concat_tables(tables))
        # Generic row fallback. A union-of-names pass plus row.get(...) was as fast as the
        # earlier DictRow-specific special case and simpler to keep correct for mixed rows.
        return cls(_table_from_rows(rows))

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


# everything below is for fast from_rows
def _table_from_rows(rows: Sequence[Row]) -> pa.Table:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name == SHARD_ID_COLUMN or name in seen:
                continue
            seen.add(name)
            names.append(name)
    columns = {name: [row.get(name) for row in rows] for name in names}
    if any(row.shard_id is not None for row in rows):
        columns[SHARD_ID_COLUMN] = [row.shard_id for row in rows]
    return pa.table(columns)


def _sorted_arrow_rows(rows: Sequence[Row]) -> Sequence[Row]:
    return sorted(
        rows,
        key=lambda row: (
            _base_arrow_row(row).tabular.tabular_id,
            row.shard_id or "",
            _base_arrow_row(row).row_idx,
        ),
    )


def _arrow_tables_from_rows(rows: Sequence[Row]) -> list[pa.Table]:
    tables: list[pa.Table] = []
    # Pristine Arrow batches do not need any overlay bookkeeping; skipping it improved the
    # plain Arrow cases without affecting patched batches.
    changed_specs = (
        _overlay_changed_specs(rows)
        if any(isinstance(row, _OverlayRow) for row in rows)
        else []
    )
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
            _arrow_table_from_group(rows[group_start:group_end], changed_specs)
        )
        group_start = group_end
    return tables


def _arrow_table_from_group(
    rows: Sequence[Row], changed_specs: list[tuple[str, pa.DataType]]
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

    changed_columns = _overlay_changes_by_column(rows)
    for name, value_type in changed_specs:
        changes = changed_columns.get(name, [])
        if name in table.column_names:
            values = table[name].to_pylist()
        else:
            values = [None] * len(rows)
        for idx, value in changes:
            values[idx] = value
        column = pa.array(values, type=value_type)
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, column)
        else:
            table = table.append_column(name, column)

    shard_id = rows[0].shard_id
    if shard_id is not None:
        shard_col = pa.array([shard_id] * len(rows), type=pa.string())
        if SHARD_ID_COLUMN in table.column_names:
            table = table.set_column(
                table.column_names.index(SHARD_ID_COLUMN),
                SHARD_ID_COLUMN,
                shard_col,
            )
        else:
            table = table.append_column(SHARD_ID_COLUMN, shard_col)
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


def _overlay_changed_specs(rows: Sequence[Row]) -> list[tuple[str, pa.DataType]]:
    specs: list[tuple[str, pa.DataType]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, _OverlayRow):
            continue
        for name in row.patch:
            if name in seen:
                continue
            seen.add(name)
            specs.append((name, pa.scalar(row.patch[name]).type))
        for name in row.deleted:
            if name in seen:
                continue
            seen.add(name)
            specs.append((name, _base_arrow_row(row).tabular.table[name].type))
    return specs


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
