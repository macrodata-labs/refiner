from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TypeAlias

import pyarrow as pa

from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import Shard
from refiner.worker.metrics.api import log_throughput

_INTERNAL_SHARD_ID_KEY = "__shard_id"
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
        for unit in self.read_shard(shard):
            rows = _unit_num_rows(unit)
            if rows > 0:
                log_throughput("rows_read", rows, shard_id=shard.id, unit="rows")
            yield _with_shard_id(unit, shard.id)

    def read(self) -> Iterator[SourceUnit]:
        for shard in self.list_shards():
            yield from self.iter_shard_units(shard)

    def describe(self) -> dict[str, Any]:
        """Optional source metadata for planning/observability."""
        return {}


__all__ = ["BaseSource"]


def _unit_num_rows(unit: SourceUnit) -> int:
    if isinstance(unit, Row):
        return 1
    if isinstance(unit, Tabular):
        return int(unit.num_rows)
    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")


def _with_shard_id(unit: SourceUnit, shard_id: str) -> SourceUnit:
    if isinstance(unit, Row):
        return unit.update(**{_INTERNAL_SHARD_ID_KEY: shard_id})

    if isinstance(unit, Tabular):
        table = unit.table
        if table.num_rows == 0:
            return unit

        shard_col = pa.array([shard_id] * int(table.num_rows), type=pa.string())
        names = table.schema.names
        if _INTERNAL_SHARD_ID_KEY in names:
            idx = table.schema.get_field_index(_INTERNAL_SHARD_ID_KEY)
            return unit.with_table(
                table.set_column(idx, _INTERNAL_SHARD_ID_KEY, shard_col)
            )
        return unit.with_table(table.append_column(_INTERNAL_SHARD_ID_KEY, shard_col))

    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")
