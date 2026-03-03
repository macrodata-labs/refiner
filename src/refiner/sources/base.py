from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from refiner.ledger.shard import Shard


class BaseSource(ABC):
    """Base class for pipeline sources."""

    name: str

    @abstractmethod
    def list_shards(self) -> list[Shard]:
        raise NotImplementedError

    @abstractmethod
    def read_shard(self, shard: Shard) -> Iterator[Any]:
        raise NotImplementedError

    def read(self) -> Iterator[Any]:
        for shard in self.list_shards():
            yield from self.read_shard(shard)

    def describe(self) -> dict[str, Any]:
        """Optional source metadata for planning/observability."""
        return {}


__all__ = ["BaseSource"]
