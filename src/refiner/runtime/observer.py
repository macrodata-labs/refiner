import socket
from dataclasses import dataclass
from logging import getLogger

from refiner.ledger.shard import Shard
from refiner.platform.observer_client import ObserverClient, WorkerConfig
from refiner.runtime.errors import UserMetricsFlushError
from refiner.runtime.metrics_context import NOOP_USER_METRICS_EMITTER, UserMetricsEmitter

logger = getLogger("refiner.observer")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerObserverContext:
    """Context for observer callbacks. worker_id must be a UUID v7 string."""

    job_id: str
    stage_index: int
    worker_id: str  # UUID v7
    host: str
    config: WorkerConfig

    @classmethod
    def from_runtime(
        cls,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        host: str | None = None,
        config: WorkerConfig | None = None,
    ) -> "WorkerObserverContext":
        return cls(
            job_id=job_id,
            stage_index=stage_index,
            worker_id=worker_id,
            host=host or socket.gethostname(),
            config=config or WorkerConfig.from_runtime(),
        )


class WorkerLifecycleObserver:
    def __init__(self, *, client: ObserverClient, context: WorkerObserverContext):
        self.client = client
        self.context = context
        try:
            self.otel: UserMetricsEmitter = client.worker_telemetry(
                job_id=context.job_id,
                stage_index=context.stage_index,
                worker_id=context.worker_id,
            )
        except Exception:
            self.otel = NOOP_USER_METRICS_EMITTER

    def on_worker_start(self, *, rank: int) -> None:
        del rank
        self.client.start_worker(
            job_id=self.context.job_id,
            stage_index=self.context.stage_index,
            worker_id=self.context.worker_id,
            host=self.context.host,
            config=self.context.config,
        )

    def on_shard_start(self, shard: Shard) -> None:
        self.client.start_shard(
            job_id=self.context.job_id,
            stage_index=self.context.stage_index,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
        )
        logger.info(f"Shard {shard.id} started")

    def on_shard_finish(
        self, shard: Shard, *, status: str, error: str | None = None
    ) -> None:
        if status == "completed":
            try:
                self.otel.force_flush_user_metrics()
            except Exception as e:  # noqa: BLE001 - explicit fatal path
                raise UserMetricsFlushError(
                    f"user metrics flush failed for completed shard {shard.id}"
                ) from e
        else:
            try:
                self.otel.force_flush_user_metrics()
            except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                logger.warning(
                    "telemetry force_flush_user_metrics failed: %s: %s",
                    type(e).__name__,
                    e,
                )
        self.client.finish_shard(
            job_id=self.context.job_id,
            stage_index=self.context.stage_index,
            worker_id=self.context.worker_id,
            shard_id=shard.id,
            status=status,
            error=error,
        )
        logger.info(f"Shard {shard.id} finished with status {status}")

    def on_worker_finish(self, *, status: str, error: str | None = None) -> None:
        try:
            self.otel.force_flush_resource_metrics()
        except Exception as e:  # noqa: BLE001 - fail-open observer hooks
            logger.warning(
                "telemetry force_flush_resource_metrics failed on worker finish: %s: %s",
                type(e).__name__,
                e,
            )
        self.client.finish_worker(
            job_id=self.context.job_id,
            stage_index=self.context.stage_index,
            worker_id=self.context.worker_id,
            status=status,
            error=error,
        )

    def metrics_emitter(self) -> UserMetricsEmitter:
        return self.otel
