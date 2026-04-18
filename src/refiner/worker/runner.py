from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from refiner.execution.engine import block_num_rows
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.pipeline import RefinerPipeline
from refiner.pipeline.sinks import NullSink
from refiner.services import ServiceManager
from refiner.services.discovery import collect_pipeline_services
from refiner.worker.context import logger, set_active_run_context, set_active_step_index
from refiner.worker.lifecycle import RuntimeLifecycle
from refiner.worker.metrics.emitter import (
    NOOP_USER_METRICS_EMITTER,
    UserMetricsEmitter,
)


@dataclass(frozen=True, slots=True)
class WorkerRunStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    output_rows: int = 0


class Worker:
    def __init__(
        self,
        pipeline: RefinerPipeline,
        *,
        job_id: str,
        stage_index: int,
        worker_id: str,
        worker_name: str | None = None,
        heartbeat_interval_seconds: int = 0,
        runtime_lifecycle: RuntimeLifecycle,
        user_metrics_emitter: UserMetricsEmitter | None = None,
    ):
        self.pipeline = pipeline
        self.job_id = job_id
        self.stage_index = stage_index
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.runtime_lifecycle = runtime_lifecycle
        self.user_metrics_emitter = (
            NOOP_USER_METRICS_EMITTER
            if user_metrics_emitter is None
            else user_metrics_emitter
        )
        if self.heartbeat_interval_seconds < 0:
            raise ValueError("heartbeat_interval_seconds must be >= 0")

    def run(self) -> WorkerRunStats:
        # Source-claim state.
        previous: Shard | None = None

        # Final worker stats.
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0

        # In-flight shard bookkeeping shared with the heartbeat thread.
        inflight_by_id: dict[str, Shard] = {}
        pending_rows_by_shard: dict[str, int] = {}
        source_done_shards: set[str] = set()
        inflight_lock = threading.Lock()

        # Error state: worker failures are converted into shard failures, and
        # heartbeat failures are surfaced from the background thread.
        execution_error: Exception | None = None
        heartbeat_error: Exception | None = None

        stop_heartbeat = threading.Event()

        service_client = getattr(self.runtime_lifecycle, "client", None)
        # Service manager is used to start and manage runtime services.
        service_manager = ServiceManager(
            client=service_client,
            job_id=self.job_id,
            stage_index=self.stage_index,
            worker_id=self.worker_id,
            worker_name=self.worker_name,
        )
        runtime_services = collect_pipeline_services(self.pipeline)
        sink = self.pipeline.sink or NullSink()
        sink_step_index = (
            self.pipeline._next_step_index() if self.pipeline.sink is not None else None
        )
        runtime_services_started = False

        def _heartbeat_once() -> None:
            with inflight_lock:
                snapshot = list(inflight_by_id.values())
            if not snapshot:
                return
            self.runtime_lifecycle.heartbeat(snapshot)

        def _heartbeat_loop() -> None:
            nonlocal heartbeat_error
            while not stop_heartbeat.wait(self.heartbeat_interval_seconds):
                try:
                    _heartbeat_once()
                except Exception as e:  # noqa: BLE001
                    message = str(e).strip() or type(e).__name__
                    logger.warning(
                        "heartbeat failed for worker_id={}: {}: {}",
                        self.worker_id,
                        type(e).__name__,
                        message,
                    )
                    heartbeat_error = e
                    return

        def _complete_shard(shard_id: str) -> None:
            nonlocal completed
            with inflight_lock:
                shard = inflight_by_id.get(shard_id)
                if shard is None:
                    return
            with set_active_step_index(sink_step_index):
                sink.on_shard_complete(shard_id)
            self.user_metrics_emitter.force_flush_user_metrics()
            self.runtime_lifecycle.complete(shard)
            with inflight_lock:
                inflight_by_id.pop(shard_id, None)
                source_done_shards.discard(shard_id)
            completed += 1
            logger.success(
                "shard completed shard_id={} global_ordinal={}",
                shard.id,
                shard.global_ordinal,
            )

        def _maybe_complete_shard(shard_id: str) -> None:
            with inflight_lock:
                pending = pending_rows_by_shard.get(shard_id, 0)
                source_done = shard_id in source_done_shards
            if source_done and pending == 0:
                _complete_shard(shard_id)

        def _apply_row_delta(delta: dict[str, int]) -> None:
            touched: list[str] = []
            with inflight_lock:
                for shard_id, amount in delta.items():
                    next_value = pending_rows_by_shard.get(shard_id, 0) + amount
                    if next_value <= 0:
                        pending_rows_by_shard.pop(shard_id, None)
                    else:
                        pending_rows_by_shard[shard_id] = next_value
                    touched.append(shard_id)
            for shard_id in touched:
                _maybe_complete_shard(shard_id)

        heartbeat_thread: threading.Thread | None = None
        if self.heartbeat_interval_seconds > 0:
            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                name=f"refiner-heartbeat-{self.worker_name or 'worker'}",
                daemon=True,
            )
            heartbeat_thread.start()

        def _source_rows():
            nonlocal previous, claimed, runtime_services_started
            while True:
                if heartbeat_error is not None:
                    raise RuntimeError(f"heartbeat failed: {heartbeat_error}")
                shard = self.runtime_lifecycle.claim(previous=previous)
                if shard is None:
                    logger.info(
                        "no more shards worker_id={} claimed={}",
                        self.worker_id,
                        claimed,
                    )
                    break
                claimed += 1
                rows_read = 0
                with inflight_lock:
                    inflight_by_id[shard.id] = shard
                if runtime_services and not runtime_services_started:
                    asyncio.run(service_manager.start_services(runtime_services))
                    runtime_services_started = True
                logger.info(
                    "shard claimed shard_id={} global_ordinal={} start_key={} end_key={}",
                    shard.id,
                    shard.global_ordinal,
                    shard.start_key,
                    shard.end_key,
                )
                logger.info(
                    "shard source started shard_id={} global_ordinal={}",
                    shard.id,
                    shard.global_ordinal,
                )
                source_iter = iter(self.pipeline.source.iter_shard_units(shard))
                while True:
                    with set_active_step_index(0):
                        try:
                            unit = next(source_iter)
                        except StopIteration:
                            break
                    rows = block_num_rows(unit)
                    if rows > 0:
                        rows_read += rows
                        with inflight_lock:
                            pending_rows_by_shard[shard.id] = (
                                pending_rows_by_shard.get(shard.id, 0) + rows
                            )
                    yield unit
                logger.info(
                    "shard source finished shard_id={} global_ordinal={} rows_read={}",
                    shard.id,
                    shard.global_ordinal,
                    rows_read,
                )
                with inflight_lock:
                    source_done_shards.add(shard.id)
                _maybe_complete_shard(shard.id)
                previous = shard

        with set_active_run_context(
            job_id=self.job_id,
            stage_index=self.stage_index,
            worker_id=self.worker_id,
            worker_name=self.worker_name,
            runtime_lifecycle=self.runtime_lifecycle,
            service_manager=service_manager,
            user_metrics_emitter=self.user_metrics_emitter,
        ):
            logger.info(
                "worker started job_id={} stage_index={} worker_id={}",
                self.job_id,
                self.stage_index,
                self.worker_id,
            )
            run_exception: Exception | None = None
            try:
                try:
                    for block in self.pipeline.execute(
                        _source_rows(),
                        on_shard_delta=_apply_row_delta,
                    ):
                        if heartbeat_error is not None:
                            raise RuntimeError(f"heartbeat failed: {heartbeat_error}")
                        with set_active_step_index(sink_step_index):
                            written = sink.write_block(block)
                        _apply_row_delta(
                            {
                                shard_id: -count
                                for shard_id, count in written.items()
                                if count
                            }
                        )
                        if sink.counts_output_rows:
                            output_rows += block_num_rows(block)
                except Exception as e:
                    execution_error = e
                    failed_error = str(e).strip() or type(e).__name__
                    with inflight_lock:
                        failed_shards = list(inflight_by_id.values())
                        inflight_by_id.clear()
                        pending_rows_by_shard.clear()
                        source_done_shards.clear()
                    logger.exception(
                        "worker execution failed worker_id={} claimed={} completed={} in_flight={} error={}",
                        self.worker_id,
                        claimed,
                        completed,
                        len(failed_shards),
                        failed_error,
                    )
                    self.user_metrics_emitter.force_flush_logs()
                    for shard in failed_shards:
                        logger.warning(
                            "shard failed shard_id={} global_ordinal={} error={}",
                            shard.id,
                            shard.global_ordinal,
                            failed_error,
                        )
                        self.user_metrics_emitter.force_flush_logs()
                        self.runtime_lifecycle.fail(shard, failed_error)
                        self.user_metrics_emitter.force_flush_user_metrics()
                        failed += 1
                else:
                    _heartbeat_once()
                    with inflight_lock:
                        remaining_shards = list(inflight_by_id.values())
                    if remaining_shards:
                        raise RuntimeError(
                            "worker finished with unflushed shards: "
                            + ", ".join(shard.id for shard in remaining_shards)
                        )

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
                stop_heartbeat.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=1.0)

                try:
                    with set_active_step_index(sink_step_index):
                        sink.close()
                except Exception as e:
                    if execution_error is not None or run_exception is not None:
                        logger.warning(
                            "sink close failed during worker failure handling: {}: {}",
                            type(e).__name__,
                            e,
                        )
                    else:
                        raise

                status = (
                    "failed"
                    if execution_error is not None or run_exception is not None
                    else "completed"
                )
                log_completion = (
                    logger.success if status == "completed" else logger.info
                )
                log_completion(
                    "worker finished job_id={} stage_index={} worker_id={} status={}",
                    self.job_id,
                    self.stage_index,
                    self.worker_id,
                    status,
                )
