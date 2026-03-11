from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from refiner.ledger import BaseLedger
from refiner.ledger.shard import Shard
from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline
from refiner.runtime.execution.engine import block_num_rows
from refiner.runtime.sinks import NullSink, ShardCompletionListener
from refiner.runtime.sinks.base import negate_counts
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
        self.sink = pipeline.sink if pipeline.sink is not None else NullSink()
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
        inflight: dict[str, Shard] = {}
        shard_closed: set[str] = set()
        shard_live_rows: dict[str, int] = {}
        failed_error: str | None = None
        lifecycle_client = self.lifecycle_client
        lifecycle_context = self.lifecycle_context
        user_metrics_emitter: UserMetricsEmitter = NOOP_USER_METRICS_EMITTER
        obs_logger = logger.bind(rank=self.rank)
        completion_listeners: list[ShardCompletionListener] = []
        if isinstance(self.sink, ShardCompletionListener):
            completion_listeners.append(self.sink)
        for step in self.pipeline.pipeline_steps:
            if isinstance(step, ShardCompletionListener):
                completion_listeners.append(step)

        class _CompletionHookError(RuntimeError):
            pass

        if lifecycle_client is not None:
            obs_logger.info(
                "worker started job_id={} stage_id={} worker_id={}",
                lifecycle_context.job_id,
                lifecycle_context.stage_id,
                lifecycle_context.worker_id,
            )
            try:
                telemetry_emitter = lifecycle_client.worker_telemetry(
                    job_id=lifecycle_context.job_id,
                    stage_id=lifecycle_context.stage_id,
                    worker_id=lifecycle_context.worker_id,
                )
            except Exception as e:  # noqa: BLE001 - fail-open telemetry setup
                obs_logger.warning(
                    "telemetry setup failed: {}: {}",
                    type(e).__name__,
                    e,
                )
            else:
                user_metrics_emitter = telemetry_emitter

        def _complete_shard(shard_id: str) -> None:
            nonlocal completed
            shard = inflight.get(shard_id)
            if shard is None:
                return
            for listener in completion_listeners:
                listener.on_shard_complete(shard_id)
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
            inflight.pop(shard_id, None)
            completed += 1

        def _maybe_complete_shard(shard_id: str) -> None:
            # A shard can complete only when:
            # 1) the source has been fully consumed for that shard (closed),
            # 2) live row count reaches zero after all step/sink deltas.
            if shard_id not in inflight:
                return
            if shard_id not in shard_closed:
                return
            if shard_live_rows.get(shard_id, 0) != 0:
                return
            try:
                _complete_shard(shard_id)
            except Exception as e:  # noqa: BLE001 - preserve existing surface behavior
                raise _CompletionHookError(str(e)) from e

        def _apply_shard_delta(delta: dict[str, int]) -> None:
            # Single place where live-row accounting mutates.
            # Positive delta: rows entered/expanded inside stage.
            # Negative delta: rows filtered/consumed/flushed at sink.
            for shard_id, change in delta.items():
                if change == 0:
                    continue
                next_value = shard_live_rows.get(shard_id, 0) + int(change)
                if next_value == 0:
                    shard_live_rows.pop(shard_id, None)
                else:
                    shard_live_rows[shard_id] = next_value
                _maybe_complete_shard(shard_id)

        def _source_rows():
            nonlocal previous, claimed
            while True:
                shard = self.ledger.claim(previous=previous)
                if shard is None:
                    break
                claimed += 1
                inflight[shard.id] = shard
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
                        obs_logger.warning(
                            "lifecycle reporting failed: {}: {}",
                            type(e).__name__,
                            e,
                        )
                with set_active_step_index(0):
                    for unit in self.pipeline.source.iter_shard_units(shard):
                        rows = block_num_rows(unit)
                        if rows > 0:
                            # Source rows are now live in this stage.
                            _apply_shard_delta({shard.id: rows})
                        yield unit
                shard_closed.add(shard.id)
                _maybe_complete_shard(shard.id)
                previous = shard

        with set_active_user_metrics_emitter(user_metrics_emitter):
            run_exception: Exception | None = None
            try:
                try:
                    for block in self.pipeline.execute(
                        _source_rows(),
                        on_shard_delta=_apply_shard_delta,
                    ):
                        # Worker runtime is where sink side effects happen.
                        counts = self.sink.write_block(block)
                        if counts:
                            _apply_shard_delta(negate_counts(counts))
                        produced = sum(int(v) for v in counts.values())
                        if produced <= 0:
                            continue
                        output_rows += produced
                        output_rows_since_hb += produced
                        while output_rows_since_hb >= self.heartbeat_every_rows:
                            for shard in inflight.values():
                                self.ledger.heartbeat(shard)
                            output_rows_since_hb -= self.heartbeat_every_rows
                except _CompletionHookError:
                    raise
                except Exception as e:
                    failed_error = str(e)
                    for shard in list(inflight.values()):
                        self.ledger.fail(shard, str(e))
                        if lifecycle_client is not None:
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
                        user_metrics_emitter.force_flush_user_metrics()
                        failed += 1
                    inflight.clear()
                else:
                    for shard_id in list(inflight.keys()):
                        _complete_shard(shard_id)

                return WorkerRunStats(
                    claimed=claimed,
                    completed=completed,
                    failed=failed,
                    output_rows=output_rows,
                )
            except Exception as e:
                run_exception = e
                raise
            finally:
                sink_close_error: Exception | None = None
                try:
                    self.sink.close()
                except Exception as e:
                    sink_close_error = e

                if lifecycle_client is not None:
                    status = (
                        "failed"
                        if failed_error is not None
                        or run_exception is not None
                        or sink_close_error is not None
                        else "completed"
                    )
                    error = failed_error
                    if error is None and run_exception is not None:
                        error = str(run_exception)
                    if error is None and sink_close_error is not None:
                        error = str(sink_close_error)
                    try:
                        lifecycle_client.report_worker_finished(
                            job_id=lifecycle_context.job_id,
                            stage_id=lifecycle_context.stage_id,
                            worker_id=lifecycle_context.worker_id,
                            status=status,
                            error=error,
                        )
                        obs_logger.info(
                            "worker finished job_id={} stage_id={} worker_id={} status={}",
                            lifecycle_context.job_id,
                            lifecycle_context.stage_id,
                            lifecycle_context.worker_id,
                            status,
                        )
                    except Exception as e:  # noqa: BLE001 - fail-open observer hooks
                        obs_logger.warning(
                            "lifecycle reporting failed: {}: {}",
                            type(e).__name__,
                            e,
                        )

                had_primary_failure = (
                    failed_error is not None or run_exception is not None
                )
                if sink_close_error is not None and had_primary_failure:
                    obs_logger.warning(
                        "sink close failed during worker failure handling: {}: {}",
                        type(sink_close_error).__name__,
                        sink_close_error,
                    )

                if (
                    failed_error is None
                    and run_exception is None
                    and sink_close_error is None
                ):
                    user_metrics_emitter.shutdown()
                else:
                    try:
                        user_metrics_emitter.shutdown()
                    except Exception as e:  # noqa: BLE001 - do not mask primary failure
                        obs_logger.warning(
                            "telemetry shutdown failed during worker failure handling: {}: {}",
                            type(e).__name__,
                            e,
                        )
                if sink_close_error is not None and not had_primary_failure:
                    raise sink_close_error


__all__ = ["Worker", "WorkerRunStats", "WorkerLifecycleContext"]
