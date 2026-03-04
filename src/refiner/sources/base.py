from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from refiner.sources.row import Row
from refiner.ledger.shard import Shard
from refiner.metrics import log_throughput

_INTERNAL_SHARD_ID_KEY = "__shard_id"


class BaseSource(ABC):
    """Base class for pipeline sources."""

    name: str

    @abstractmethod
    def list_shards(self) -> list[Shard]:
        raise NotImplementedError

    @abstractmethod
    def read_shard(self, shard: Shard) -> Iterator[Row]:
        raise NotImplementedError

    def iter_shard_rows(self, shard: Shard) -> Iterator[Row]:
        for row in self.read_shard(shard):
            log_throughput("rows_read", 1, shard_id=shard.id, unit="rows")
            yield row.update(**{_INTERNAL_SHARD_ID_KEY: shard.id})

    def read(self) -> Iterator[Row]:
        for shard in self.list_shards():
            yield from self.iter_shard_rows(shard)

    def describe(self) -> dict[str, Any]:
        """Optional source metadata for planning/observability."""
        return {}


__all__ = ["BaseSource"]
