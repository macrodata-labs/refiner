from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TypeAlias

import pyarrow as pa

from refiner.pipeline.data.tabular import Tabular, repeat_scalar, set_or_append_column
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN, SOURCE_ROW_ID_COLUMN, Shard
from refiner.worker.metrics.api import log_throughput

SourceUnit: TypeAlias = Row | Tabular


class BaseSource(ABC):
    """Base class for pipeline sources."""

    name: str

    @abstractmethod
    def list_shards(self) -> list[Shard]:
        raise NotImplementedError

    @abstractmethod
    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        raise NotImplementedError

    def iter_shard_units(self, shard: Shard) -> Iterator[SourceUnit]:
        next_source_row_id = 0
        for unit in self.read_shard(shard):
            rows = _unit_num_rows(unit)
            if rows > 0:
                log_throughput("rows_read", rows, shard_id=shard.id, unit="rows")
            yield _with_source_lineage(
                unit,
                shard_id=shard.id,
                first_source_row_id=next_source_row_id,
            )
            next_source_row_id += rows

    def read(self) -> Iterator[SourceUnit]:
        for shard in self.list_shards():
            yield from self.iter_shard_units(shard)

    @property
    def schema(self) -> pa.Schema | None:
        return None

    def describe(self) -> dict[str, Any]:
        """Optional source metadata for planning/observability."""
        return {}

    def required_refiner_extras(self) -> tuple[str, ...]:
        """macrodata-refiner extras required by this source."""
        return tuple(
            sorted(
                {
                    *self._declared_refiner_extras(),
                    *self._io_refiner_extras(),
                }
            )
        )

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        """Feature extras declared by this source."""
        return ()

    def _io_refiner_extras(self) -> tuple[str, ...]:
        """Storage extras required by this source's normalized IO handles."""
        return ()


__all__ = ["BaseSource"]


def _unit_num_rows(unit: SourceUnit) -> int:
    if isinstance(unit, Row):
        return 1
    if isinstance(unit, Tabular):
        return int(unit.num_rows)
    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")


def _with_source_lineage(
    unit: SourceUnit,
    *,
    shard_id: str,
    first_source_row_id: int,
) -> SourceUnit:
    if isinstance(unit, Row):
        identity: dict[str, str | int] = {SHARD_ID_COLUMN: shard_id}
        if unit.source_row_id is None:
            identity[SOURCE_ROW_ID_COLUMN] = first_source_row_id
        return unit.update(identity)

    if isinstance(unit, Tabular):
        table = unit.table
        if table.num_rows == 0:
            return unit

        shard_col = repeat_scalar(pa.scalar(shard_id, type=pa.string()), table.num_rows)
        with_shard_id = unit.with_table(
            set_or_append_column(table, SHARD_ID_COLUMN, shard_col)
        )
        if SOURCE_ROW_ID_COLUMN in with_shard_id.table.column_names:
            return with_shard_id
        return with_shard_id.with_table(
            set_or_append_column(
                with_shard_id.table,
                SOURCE_ROW_ID_COLUMN,
                pa.array(
                    range(first_source_row_id, first_source_row_id + table.num_rows),
                    type=pa.uint64(),
                ),
            )
        )

    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")
