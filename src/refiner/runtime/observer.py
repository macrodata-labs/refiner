import socket
from dataclasses import dataclass
from logging import getLogger

from refiner.ledger.shard import Shard
from refiner.platform.observer_client import ObserverClient, WorkerConfig
from refiner.platform.telemetry import NoopTelemetryEmitter, WorkerTelemetry

logger = getLogger("refiner.observer")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerObserverContext:
    """Context for observer callbacks. worker_id must be a UUID v7 string."""

    job_id: str
    stage_id: str
    worker_id: str  # UUID v7
    host: str
    config: WorkerConfig

    @classmethod
    def from_runtime(
        cls,
        *,
        job_id: str,
        stage_id: str,
        worker_id: str,
        host: str | None = None,
        config: WorkerConfig | None = None,
    ) -> "WorkerObserverContext":
        return cls(
            job_id=job_id,
            stage_id=stage_id,
            worker_id=worker_id,
            host=host or socket.gethostname(),
            config=config or WorkerConfig.from_runtime(),
        )


class WorkerLifecycleObserver:
    def __init__(self, *, client: ObserverClient, context: WorkerObserverContext):
        self.client = client
        self.context = context
        try:
            self.otel: WorkerTelemetry = client.worker_telemetry(
                job_id=context.job_id,
                stage_id=context.stage_id,
                worker_id=context.worker_id,
            )
        except Exception:
            self.otel = NoopTelemetryEmitter()

    def on_worker_start(self, *, rank: int) -> None:
        del rank
        self.client.start_worker(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            host=self.context.host,
            config=self.context.config,
        )

    def on_shard_start(self, shard: Shard) -> None:
        self.client.start_shard(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
        )
        logger.info(f"Shard {shard.id} started")

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
        logger.info(f"Shard {shard.id} finished with status {status}")

    def on_worker_finish(self, *, status: str, error: str | None = None) -> None:
        self.client.finish_worker(
            job_id=self.context.job_id,
            stage_id=self.context.stage_id,
            worker_id=self.context.worker_id,
            status=status,
            error=error,
        )

    def on_records_processed(self, count: int) -> None:
        self.otel.record_records_processed(count)
