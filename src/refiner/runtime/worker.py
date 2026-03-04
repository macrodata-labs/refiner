from __future__ import annotations

from dataclasses import dataclass
import sys

from loguru import logger

from refiner.ledger import BaseLedger
from refiner.ledger.shard import Shard
from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline
from refiner.runtime.metrics_context import (
    NOOP_USER_METRICS_EMITTER,
    UserMetricsEmitter,
    set_active_step_index,
    set_active_user_metrics_emitter,
)


@dataclass(frozen=True, slots=True)
class WorkerRunStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    output_rows: int = 0


@dataclass(frozen=True, slots=True)
class WorkerLifecycleContext:
    job_id: str
    stage_id: str
    worker_id: str
    worker_name: str | None = None


class Worker:
    def __init__(
        self,
        rank: int,
        ledger: BaseLedger,
        pipeline: RefinerPipeline,
        *,
        heartbeat_every_rows: int = 4096,
        lifecycle_client: MacrodataClient | None = None,
        lifecycle_context: WorkerLifecycleContext,
    ):
        self.rank = rank
        self.ledger = ledger
        self.pipeline = pipeline
        self.heartbeat_every_rows = heartbeat_every_rows
        self.lifecycle_client = lifecycle_client
        self.lifecycle_context = lifecycle_context

    def run(self) -> WorkerRunStats:
        if self.heartbeat_every_rows <= 0:
            raise ValueError("heartbeat_every_rows must be > 0")

        previous: Shard | None = None
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        output_rows_since_hb = 0
        inflight: list[Shard] = []
        failed_error: str | None = None
        lifecycle_client = self.lifecycle_client
        lifecycle_context = self.lifecycle_context
        user_metrics_emitter: UserMetricsEmitter = NOOP_USER_METRICS_EMITTER
        obs_logger = logger.bind(rank=self.rank)

        if lifecycle_client is not None:
            obs_logger.info(
                "worker started job_id={} stage_id={} worker_id={}",
                lifecycle_context.job_id,
                lifecycle_context.stage_id,
                lifecycle_context.worker_id,
            )
            try:
                user_metrics_emitter = lifecycle_client.worker_telemetry(
                    job_id=lifecycle_context.job_id,
                    stage_id=lifecycle_context.stage_id,
                    worker_id=lifecycle_context.worker_id,
                )
            except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                print(
                    f"[refiner] telemetry setup failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

        def _source_rows():
            nonlocal previous, claimed
            while True:
                shard = self.ledger.claim(previous=previous)
                if shard is None:
                    break
                claimed += 1
                inflight.append(shard)
                if lifecycle_client is not None:
                    try:
                        lifecycle_client.report_shard_started(
                            job_id=lifecycle_context.job_id,
                            stage_id=lifecycle_context.stage_id,
                            worker_id=lifecycle_context.worker_id,
                            shard_id=shard.id,
                        )
                        obs_logger.info(
                            "shard started job_id={} stage_id={} worker_id={} shard_id={}",
                            lifecycle_context.job_id,
                            lifecycle_context.stage_id,
                            lifecycle_context.worker_id,
                            shard.id,
                        )
                    except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                        print(
                            "[refiner] lifecycle reporting failed: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                with set_active_step_index(0):
                    yield from self.pipeline.source.iter_shard_rows(shard)
                previous = shard

        with set_active_user_metrics_emitter(user_metrics_emitter):
            while True:
                try:
                    for _ in self.pipeline.execute_rows(_source_rows()):
                        output_rows += 1
                        output_rows_since_hb += 1
                        if output_rows_since_hb % self.heartbeat_every_rows == 0:
                            for shard in inflight:
                                self.ledger.heartbeat(shard)

                    for shard in list(inflight):
                        self.ledger.heartbeat(shard)
                        if lifecycle_client is not None:
                                lifecycle_client.report_shard_finished(
                                    job_id=lifecycle_context.job_id,
                                    stage_id=lifecycle_context.stage_id,
                                    worker_id=lifecycle_context.worker_id,
                                    shard_id=shard.id,
                                    status="completed",
                                    error=None,
                                )
                                obs_logger.info(
                                    "shard finished job_id={} stage_id={} worker_id={} shard_id={} status=completed",
                                    lifecycle_context.job_id,
                                    lifecycle_context.stage_id,
                                    lifecycle_context.worker_id,
                                    shard.id,
                                )
                        user_metrics_emitter.force_flush_user_metrics()
                        self.ledger.complete(shard)
                        inflight.remove(shard)
                        completed += 1
                    inflight.clear()
                    break
                except Exception as e:
                    failed_error = str(e)
                    for shard in inflight:
                        self.ledger.fail(shard, str(e))
                        if lifecycle_client is not None:
                            try:
                                lifecycle_client.report_shard_finished(
                                    job_id=lifecycle_context.job_id,
                                    stage_id=lifecycle_context.stage_id,
                                    worker_id=lifecycle_context.worker_id,
                                    shard_id=shard.id,
                                    status="failed",
                                    error=str(e),
                                )
                                obs_logger.info(
                                    "shard finished job_id={} stage_id={} worker_id={} shard_id={} status=failed",
                                    lifecycle_context.job_id,
                                    lifecycle_context.stage_id,
                                    lifecycle_context.worker_id,
                                    shard.id,
                                )
                            except Exception as e2:  # noqa: BLE001 - fail-open observer hooks
                                print(
                                    "[refiner] lifecycle reporting failed: "
                                    f"{type(e2).__name__}: {e2}",
                                    file=sys.stderr,
                                )
                        failed += 1
                    inflight.clear()
                    previous = None
                    break

        if lifecycle_client is not None:
            try:
                lifecycle_client.report_worker_finished(
                    job_id=lifecycle_context.job_id,
                    stage_id=lifecycle_context.stage_id,
                    worker_id=lifecycle_context.worker_id,
                    status="failed" if failed_error is not None else "completed",
                    error=failed_error,
                )
                obs_logger.info(
                    "worker finished job_id={} stage_id={} worker_id={} status={}",
                    lifecycle_context.job_id,
                    lifecycle_context.stage_id,
                    lifecycle_context.worker_id,
                    "failed" if failed_error is not None else "completed",
                )
            except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                print(
                    f"[refiner] lifecycle reporting failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
        try:
            user_metrics_emitter.force_flush_user_metrics()
            user_metrics_emitter.force_flush_resource_metrics()
            user_metrics_emitter.force_flush_logs()
        finally:
            user_metrics_emitter.shutdown()

        return WorkerRunStats(
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )


__all__ = ["Worker", "WorkerRunStats", "WorkerLifecycleContext"]
