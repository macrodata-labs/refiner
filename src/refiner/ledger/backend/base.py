from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from ..shard import Shard


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    lease_seconds: int = 10 * 60
    # Exposed so callers can schedule heartbeats at a reasonable default cadence.
    heartbeat_seconds: int = 2 * 60


class BaseLedger(ABC):
    """Ledger for shard work distribution."""

    def __init__(self, *, job_id: str, worker_id: int | None, config: LedgerConfig):
        if not job_id:
            raise ValueError("job_id must be non-empty")
        # worker_id is optional so a coordinator can seed shards without pretending to be a worker.
        self.worker_id = int(worker_id) if worker_id is not None else None
        self.job_id = str(job_id)
        self.config = config

    def _require_worker_id(self) -> int:
        if self.worker_id is None:
            raise ValueError(
                "worker_id is required for this operation (initialize the ledger with worker_id=...)"
            )
        return self.worker_id

    def _require_worker_id_str(self) -> str:
        return str(self._require_worker_id())

    @abstractmethod
    def seed_shards(self, shards: Iterable[Shard]) -> None:
        """Overwrite the shard set for this run (seed the queue)."""
        raise NotImplementedError

    @abstractmethod
    def claim(self, previous: Shard | None = None) -> Shard | None:
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


__all__ = ["BaseLedger", "LedgerConfig"]
