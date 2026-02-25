from __future__ import annotations

import socket
from dataclasses import dataclass

from refiner.ledger.shard import Shard
from refiner.platform.observer_client import ObserverClient


@dataclass(frozen=True, slots=True)
class WorkerObserverContext:
    job_id: str
    stage_id: str
    worker_id: str


class WorkerLifecycleObserver:
    def __init__(self, *, client: ObserverClient, context: WorkerObserverContext):
        self.client = client
        self.context = context

    def on_worker_start(self, *, rank: int) -> None:
        del rank
        try:
            host = socket.gethostname()
        except Exception:
            host = None
        self.client.start_worker(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            host=host,
        )

    def on_shard_start(self, shard: Shard) -> None:
        self.client.start_shard(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
        )

    def on_shard_finish(
        self, shard: Shard, *, status: str, error: str | None = None
    ) -> None:
        self.client.finish_shard(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
            status=status,
            error=error,
        )

    def on_worker_finish(self, *, status: str, error: str | None = None) -> None:
        self.client.finish_worker(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            status=status,
            error=error,
        )
