from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from refiner.io import DataFolder
from refiner.pipeline.data.row import ArrowRowView, Row
from refiner.pipeline.data.tabular import Tabular
from refiner.robotics.lerobot_format.metadata.metadata import LeRobotMetadata

if TYPE_CHECKING:
    from refiner.robotics.lerobot_format.row import LeRobotRow

_ROW_INDEX_COLUMN = "__lerobot_row_index"


@dataclass(init=False, frozen=True, slots=True)
class LeRobotTabular(Tabular):
    _tabular: Tabular
    metadata_by_row: tuple[LeRobotMetadata, ...]
    frames_by_row: tuple[Sequence[Row] | Tabular, ...]
    roots_by_row: tuple[DataFolder | None, ...]

    def __init__(
        self,
        tabular: Tabular,
        *,
        metadata_by_row: tuple[LeRobotMetadata, ...],
        frames_by_row: tuple[Sequence[Row] | Tabular, ...],
        roots_by_row: tuple[DataFolder | None, ...],
    ) -> None:
        num_rows = tabular.num_rows
        if len(metadata_by_row) != num_rows:
            raise ValueError("metadata_by_row length must match tabular row count")
        if len(frames_by_row) != num_rows:
            raise ValueError("frames_by_row length must match tabular row count")
        if len(roots_by_row) != num_rows:
            raise ValueError("roots_by_row length must match tabular row count")
        object.__setattr__(self, "_tabular", tabular)
        object.__setattr__(self, "metadata_by_row", metadata_by_row)
        object.__setattr__(self, "frames_by_row", frames_by_row)
        object.__setattr__(self, "roots_by_row", roots_by_row)

    @classmethod
    def from_rows(cls, rows: Sequence[Row]) -> "LeRobotTabular":
        from refiner.robotics.lerobot_format.row import LeRobotRow

        if not all(isinstance(row, LeRobotRow) for row in rows):
            raise TypeError("LeRobotTabular.from_rows requires LeRobotRow inputs")
        lerobot_rows = tuple(cast("LeRobotRow", row) for row in rows)
        base = Tabular.from_rows(
            [
                row._row.update({_ROW_INDEX_COLUMN: idx})
                for idx, row in enumerate(lerobot_rows)
            ]
        )
        row_order = [
            int(value) for value in base.table.column(_ROW_INDEX_COLUMN).to_pylist()
        ]
        table = base.table.drop_columns([_ROW_INDEX_COLUMN])
        return cls(
            base.with_table(table),
            metadata_by_row=tuple(lerobot_rows[idx].metadata for idx in row_order),
            frames_by_row=tuple(lerobot_rows[idx].frames for idx in row_order),
            roots_by_row=tuple(lerobot_rows[idx].root for idx in row_order),
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

    def with_table(self, table: pa.Table) -> "LeRobotTabular":
        if table.num_rows != len(self.metadata_by_row):
            raise ValueError(
                "LeRobotTabular side data length must match table row count"
            )
        return LeRobotTabular(
            self._tabular.with_table(table),
            metadata_by_row=self.metadata_by_row,
            frames_by_row=self.frames_by_row,
            roots_by_row=self.roots_by_row,
        )

    def to_rows(self) -> list[Row]:
        from refiner.robotics.lerobot_format.row import LeRobotRow

        rows: list[Row] = []
        for row_idx in range(self.table.num_rows):
            shard_id = None
            if self.shard_idx is not None:
                shard = self.columns[self.shard_idx][row_idx].as_py()
                shard_id = shard if isinstance(shard, str) else None
            rows.append(
                LeRobotRow(
                    _row=ArrowRowView(
                        tabular=self,
                        row_idx=row_idx,
                        shard_id=shard_id,
                    ),
                    metadata=self.metadata_by_row[row_idx],
                    frames=self.frames_by_row[row_idx],
                    root=self.roots_by_row[row_idx],
                )
            )
        return rows


__all__ = ["LeRobotTabular"]
