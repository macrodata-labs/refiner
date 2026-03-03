from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from refiner.sources.row import Row
from refiner.ledger.shard import Shard
from refiner.metrics import log_counter


class BaseSource(ABC):
    """Base class for pipeline sources."""

    name: str

    @abstractmethod
    def list_shards(self) -> list[Shard]:
        raise NotImplementedError

    @abstractmethod
    def read_shard(self, shard: Shard) -> Iterator[Any]:
        raise NotImplementedError

    def iter_shard_rows(self, shard: Shard) -> Iterator[Row]:
        for row in self.read_shard(shard):
            log_counter("rows_read", 1, shard_id=shard.id)
            yield row.update(shard_id=shard.id)

    def read(self) -> Iterator[Any]:
        for shard in self.list_shards():
            yield from self.iter_shard_rows(shard)

    def describe(self) -> dict[str, Any]:
        """Optional source metadata for planning/observability."""
        return {}


__all__ = ["BaseSource"]
