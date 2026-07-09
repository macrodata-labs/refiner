from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.robotics.row import _RoboticsRowSpec, _RoboticsRowView
from refiner.video import VideoSource

_MISSING = object()
_SIDE_DATA_ROW_INDEX = "__robotics_side_data_row_index"


@dataclass(init=False, frozen=True, slots=True)
class RoboticsTabular(Tabular):
    _tabular: Tabular
    spec: _RoboticsRowSpec
    side_data: Mapping[str, tuple[Any, ...]]

    def __init__(
        self,
        tabular: Tabular,
        *,
        spec: _RoboticsRowSpec,
        side_data: Mapping[str, tuple[Any, ...]] | None = None,
    ) -> None:
        object.__setattr__(self, "_tabular", tabular)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "side_data", side_data or {})

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Row],
        *,
        schema: pa.Schema | None = None,
    ) -> "RoboticsTabular | Tabular":
        if not rows:
            raise ValueError("RoboticsTabular.from_rows requires at least one row")
        if not all(isinstance(row, _RoboticsRowView) for row in rows):
            return Tabular.from_rows(rows, schema=schema)
        robotics_rows = tuple(cast(_RoboticsRowView, row) for row in rows)
        spec = robotics_rows[0]._spec
        if any(row._spec != spec for row in robotics_rows):
            return Tabular.from_rows(rows, schema=schema)
        raw_rows = [row._row for row in robotics_rows]
        side_data = _side_data(raw_rows)
        row_index_key = _row_index_key(raw_rows) if side_data else None
        table_rows = [
            _without_side_data(row, tuple(side_data)).update({row_index_key: idx})
            if row_index_key is not None
            else row
            for idx, row in enumerate(raw_rows)
        ]
        table = Tabular.from_rows(table_rows, schema=schema)
        if row_index_key is not None:
            row_order = [int(idx) for idx in table.column(row_index_key).to_pylist()]
            table = table.with_table(table.table.drop([row_index_key]))
            side_data = {
                key: tuple(values[idx] for idx in row_order)
                for key, values in side_data.items()
            }
        return cls(
            table,
            spec=spec,
            side_data=side_data,
        )

    @property
    def unit(self) -> pa.Table:
        return self._tabular.unit

    @property
    def table(self) -> pa.Table:
        return self._tabular.table

    @property
    def tabular_id(self) -> int:
        return self._tabular.tabular_id

    @property
    def names(self) -> tuple[str, ...]:
        return self._tabular.names

    @property
    def columns(self) -> tuple[pa.Array | pa.ChunkedArray, ...]:
        return self._tabular.columns

    @property
    def index_by_name(self) -> dict[str, int]:
        return self._tabular.index_by_name

    @property
    def shard_idx(self) -> int | None:
        return self._tabular.shard_idx

    @property
    def needs_row_indices(self) -> bool:
        return bool(self.side_data)

    def with_table(
        self,
        table: pa.Table,
        *,
        row_indices: Sequence[int] | None = None,
    ) -> "RoboticsTabular":
        side_data = self.side_data
        if side_data and row_indices is not None:
            side_data = {
                key: tuple(values[int(idx)] for idx in row_indices)
                for key, values in side_data.items()
            }
        elif side_data and table.num_rows != self._tabular.num_rows:
            raise ValueError(
                "RoboticsTabular side data length must match table row count"
            )
        return RoboticsTabular(
            self._tabular.with_table(table, row_indices=row_indices),
            spec=self.spec,
            side_data=side_data,
        )

    def __iter__(self) -> Iterator[Row]:
        for idx, row in enumerate(self._tabular):
            if self.side_data:
                row = row.update(
                    {
                        key: value
                        for key, values in self.side_data.items()
                        if (value := values[idx]) is not _MISSING
                    }
                )
            yield cast(Row, self.spec.wrap(row))

    def to_rows(self) -> list[Row]:
        return list(self)


def _side_data(rows: Sequence[Row]) -> dict[str, tuple[Any, ...]]:
    keys = tuple(dict.fromkeys(key for row in rows for key in row))
    side_keys = {
        key
        for key in keys
        if any(_is_side_data_value(row.get(key, _MISSING)) for row in rows)
    }
    return {key: tuple(row.get(key, _MISSING) for row in rows) for key in side_keys}


def _without_side_data(row: Row, side_keys: Sequence[str]) -> Row:
    if not side_keys:
        return row
    return row.drop(*side_keys)


def _row_index_key(rows: Sequence[Row]) -> str:
    key = _SIDE_DATA_ROW_INDEX
    while any(key in row for row in rows):
        key = f"_{key}"
    return key


def _is_side_data_value(value: Any) -> bool:
    return value is not _MISSING and (
        isinstance(value, Tabular | VideoSource | pa.Array | pa.ChunkedArray)
        or bool(getattr(value, "__refiner_side_data__", False))
    )


__all__ = ["RoboticsTabular"]
