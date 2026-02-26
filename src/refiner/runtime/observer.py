from __future__ import annotations

import socket
from dataclasses import dataclass

from refiner.ledger.shard import Shard
from refiner.platform.client import MacrodataClient


@dataclass(frozen=True, slots=True)
class WorkerObserverContext:
    job_id: str
    stage_id: str
    worker_id: str


class WorkerLifecycleObserver:
    def __init__(self, *, client: MacrodataClient, context: WorkerObserverContext):
        self.client = client
        self.context = context

    def on_worker_start(self, *, rank: int) -> None:
        del rank
        try:
            host = socket.gethostname()
        except Exception:
            host = None
        self.client.report_worker_started(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            host=host,
        )

    def on_shard_start(self, shard: Shard) -> None:
        self.client.report_shard_started(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
        )

    def on_shard_finish(
        self, shard: Shard, *, status: str, error: str | None = None
    ) -> None:
        self.client.report_shard_finished(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
            status=status,
            error=error,
        )

    def on_worker_finish(self, *, status: str, error: str | None = None) -> None:
        self.client.report_worker_finished(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            status=status,
            error=error,
        )
