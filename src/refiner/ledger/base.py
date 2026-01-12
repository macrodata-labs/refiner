from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from refiner.readers.base import Shard

if TYPE_CHECKING:
    from .backend.base import LedgerConfig
else:
    # Import at runtime to avoid circular dependency
    from .backend.base import LedgerConfig


class BaseLedger(ABC):
    """A minimal shard work ledger.

    This is intentionally small: it is only responsible for shard claiming and completion
    bookkeeping, not execution orchestration.
    """

    def __init__(self, *, worker_id: str, run_id: str, config: LedgerConfig):
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.worker_id = str(worker_id)
        self.run_id = str(run_id)
        self.config = config

    @abstractmethod
    def enqueue(self, shards: Iterable[Shard]) -> None:
        raise NotImplementedError

    @abstractmethod
    def claim(self) -> Shard | None:
        raise NotImplementedError

    @abstractmethod
    def heartbeat(self, shard: Shard) -> None:
        raise NotImplementedError

    @abstractmethod
    def complete(self, shard: Shard) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail(self, shard: Shard, error: str | None = None) -> None:
        raise NotImplementedError


__all__ = ["BaseLedger"]
