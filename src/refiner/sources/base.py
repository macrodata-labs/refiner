from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from refiner.sources.row import Row
from refiner.runtime.types import SourceUnit
from refiner.ledger.shard import Shard
from refiner.ledger.shard_tracking import SHARD_ID_COLUMN
from refiner.metrics import log_throughput

_INTERNAL_SHARD_ID_KEY = SHARD_ID_COLUMN


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
    if isinstance(unit, pa.RecordBatch):
        return int(unit.num_rows)
    if isinstance(unit, pa.Table):
        return int(unit.num_rows)
    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")


def _with_shard_id(unit: SourceUnit, shard_id: str) -> SourceUnit:
    if isinstance(unit, Row):
        return unit.with_shard_id(shard_id)

    if isinstance(unit, (pa.RecordBatch, pa.Table)):
        if unit.num_rows == 0:
            return unit

        shard_col = pa.array([shard_id] * int(unit.num_rows), type=pa.string())
        names = unit.schema.names
        if _INTERNAL_SHARD_ID_KEY in names:
            idx = unit.schema.get_field_index(_INTERNAL_SHARD_ID_KEY)
            return unit.set_column(idx, _INTERNAL_SHARD_ID_KEY, shard_col)
        return unit.append_column(_INTERNAL_SHARD_ID_KEY, shard_col)

    raise TypeError(f"Unsupported source unit type: {type(unit)!r}")
