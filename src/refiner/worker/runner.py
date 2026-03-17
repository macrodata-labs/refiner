from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from refiner.execution.engine import block_num_rows
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.pipeline import RefinerPipeline
from refiner.pipeline.sinks import NullSink
from refiner.worker.context import RunHandle
from refiner.worker.context import set_active_run_context, set_active_step_index
from refiner.worker.lifecycle import (
    LocalRuntimeLifecycle,
    PlatformRuntimeLifecycle,
    RuntimeLifecycle,
)
from refiner.worker.metrics.context import (
    NOOP_USER_METRICS_EMITTER,
    UserMetricsEmitter,
    set_active_user_metrics_emitter,
)
from refiner.worker.metrics.otel import OtelTelemetryEmitter


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
        run_handle: RunHandle,
        heartbeat_interval_seconds: int = 30,
        local_workdir: str | None = None,
    ):
        self.pipeline = pipeline
        self.run_handle = run_handle
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.local_workdir = local_workdir
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")

    def _start_platform_session(self) -> tuple[PlatformRuntimeLifecycle, RunHandle]:
        if self.run_handle.client is None:
            raise ValueError("platform runtime requires a run with a client")
        try:
            host = socket.gethostname()
        except Exception:
            host = None
        started_resp = self.run_handle.client.report_worker_started(
            job_id=self.run_handle.job_id,
            stage_index=self.run_handle.stage_index,
            host=host,
            worker_name=self.run_handle.worker_name,
        )
        run = self.run_handle.with_worker(worker_id=started_resp.worker_id)
        runtime_lifecycle = PlatformRuntimeLifecycle(run=run)
        return runtime_lifecycle, run

    def _start_local_session(self) -> tuple[LocalRuntimeLifecycle, RunHandle]:
        run = self.run_handle.with_worker(worker_id=uuid4().hex[:12])
        runtime_lifecycle = LocalRuntimeLifecycle(
            run=run,
            workdir=self.local_workdir,
        )
        return runtime_lifecycle, run

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

        # Error state: completion failures are re-raised directly, execution
        # failures are converted into shard failures, and heartbeat failures
        # are surfaced from the background thread.
        completion_error: Exception | None = None
        execution_error: Exception | None = None
        heartbeat_error: Exception | None = None

        # Runtime services.
        user_metrics_emitter: UserMetricsEmitter = NOOP_USER_METRICS_EMITTER
        obs_logger = logger.bind(worker_name=self.run_handle.worker_name)
        stop_heartbeat = threading.Event()

        if self.run_handle.client is not None:
            # platform
            runtime_lifecycle, self.run_handle = self._start_platform_session()
            client = self.run_handle.client
            if client is None:
                raise ValueError("platform runtime requires a client")
            try:
                telemetry_emitter = OtelTelemetryEmitter(
                    base_url=client.base_url,
                    api_key=client.api_key,
                    job_id=self.run_handle.job_id,
                    stage_index=self.run_handle.stage_index,
                    worker_id=self.run_handle.worker_id or "",
                )
            except Exception as e:  # noqa: BLE001
                obs_logger.warning(
                    "telemetry setup failed: {}: {}",
                    type(e).__name__,
                    e,
                )
            else:
                user_metrics_emitter = telemetry_emitter
        else:
            # local mode
            runtime_lifecycle, self.run_handle = self._start_local_session()
        runtime_name = "platform" if self.run_handle.client is not None else "file"
        obs_logger.info(
            "worker started job_id={} stage_index={} worker_id={} runtime={}",
            self.run_handle.job_id,
            self.run_handle.stage_index,
            self.run_handle.worker_id,
            runtime_name,
        )
        sink = self.pipeline.sink or NullSink()

        def _heartbeat_once() -> None:
            with inflight_lock:
                snapshot = list(inflight_by_id.values())
            if not snapshot:
                return
            runtime_lifecycle.heartbeat(snapshot)

        def _heartbeat_loop() -> None:
            nonlocal heartbeat_error
            while not stop_heartbeat.wait(self.heartbeat_interval_seconds):
                try:
                    _heartbeat_once()
                except Exception as e:  # noqa: BLE001
                    message = str(e).strip() or type(e).__name__
                    obs_logger.warning(
                        "heartbeat failed for worker_id={}: {}: {}",
                        self.run_handle.worker_id,
                        type(e).__name__,
                        message,
                    )
                    heartbeat_error = e
                    return

        def _complete_shard(shard_id: str) -> None:
            nonlocal completed, completion_error
            with inflight_lock:
                shard = inflight_by_id.pop(shard_id, None)
                source_done_shards.discard(shard_id)
                if shard is None:
                    return
            try:
                sink.on_shard_complete(shard_id)
                user_metrics_emitter.force_flush_user_metrics()
                runtime_lifecycle.complete(shard)
            except Exception as err:  # noqa: BLE001
                completion_error = err
                raise
            completed += 1
            obs_logger.info(
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

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"refiner-heartbeat-{self.run_handle.worker_name or 'worker'}",
            daemon=True,
        )
        heartbeat_thread.start()

        def _source_rows():
            nonlocal previous, claimed
            while True:
                if heartbeat_error is not None:
                    raise RuntimeError(f"heartbeat failed: {heartbeat_error}")
                shard = runtime_lifecycle.claim(previous=previous)
                if shard is None:
                    obs_logger.info(
                        "no more shards worker_id={} claimed={}",
                        self.run_handle.worker_id,
                        claimed,
                    )
                    break
                claimed += 1
                rows_read = 0
                with inflight_lock:
                    inflight_by_id[shard.id] = shard
                obs_logger.info(
                    "shard claimed shard_id={} global_ordinal={} start_key={} end_key={}",
                    shard.id,
                    shard.global_ordinal,
                    shard.start_key,
                    shard.end_key,
                )
                with set_active_step_index(0):
                    obs_logger.info(
                        "shard source started shard_id={} global_ordinal={}",
                        shard.id,
                        shard.global_ordinal,
                    )
                    for unit in self.pipeline.source.iter_shard_units(shard):
                        rows = block_num_rows(unit)
                        if rows > 0:
                            rows_read += rows
                            with inflight_lock:
                                pending_rows_by_shard[shard.id] = (
                                    pending_rows_by_shard.get(shard.id, 0) + rows
                                )
                        yield unit
                obs_logger.info(
                    "shard source finished shard_id={} global_ordinal={} rows_read={}",
                    shard.id,
                    shard.global_ordinal,
                    rows_read,
                )
                with inflight_lock:
                    source_done_shards.add(shard.id)
                _maybe_complete_shard(shard.id)
                previous = shard

        with (
            set_active_user_metrics_emitter(user_metrics_emitter),
            set_active_run_context(
                run_handle=self.run_handle,
                runtime_lifecycle=runtime_lifecycle,
            ),
        ):
            run_exception: Exception | None = None
            try:
                try:
                    for block in self.pipeline.execute(
                        _source_rows(),
                        on_shard_delta=_apply_row_delta,
                    ):
                        if heartbeat_error is not None:
                            raise RuntimeError(f"heartbeat failed: {heartbeat_error}")
                        written = sink.write_block(block)
                        _apply_row_delta(
                            {
                                shard_id: -count
                                for shard_id, count in written.items()
                                if count
                            }
                        )
                        output_rows += block_num_rows(block)
                except Exception as e:
                    if completion_error is e:
                        raise
                    execution_error = e
                    failed_error = str(e).strip() or type(e).__name__
                    with inflight_lock:
                        failed_shards = list(inflight_by_id.values())
                        inflight_by_id.clear()
                        pending_rows_by_shard.clear()
                        source_done_shards.clear()
                    obs_logger.exception(
                        "worker execution failed worker_id={} claimed={} completed={} in_flight={} error={}",
                        self.run_handle.worker_id,
                        claimed,
                        completed,
                        len(failed_shards),
                        failed_error,
                    )
                    user_metrics_emitter.force_flush_logs()
                    for shard in failed_shards:
                        obs_logger.warning(
                            "shard failed shard_id={} global_ordinal={} error={}",
                            shard.id,
                            shard.global_ordinal,
                            failed_error,
                        )
                        user_metrics_emitter.force_flush_logs()
                        runtime_lifecycle.fail(shard, failed_error)
                        user_metrics_emitter.force_flush_user_metrics()
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
                heartbeat_thread.join(timeout=1.0)
                sink.close()

                if self.run_handle.client is not None:
                    current_error = execution_error or run_exception
                    status = "failed" if current_error is not None else "completed"
                    error = None
                    if current_error is not None:
                        error = (
                            str(current_error).strip() or type(current_error).__name__
                        )
                    try:
                        self.run_handle.client.report_worker_finished(
                            job_id=self.run_handle.job_id,
                            stage_index=self.run_handle.stage_index,
                            worker_id=self.run_handle.worker_id or "",
                            status=status,
                            error=error,
                        )
                        obs_logger.info(
                            "worker finished job_id={} stage_index={} worker_id={} status={}",
                            self.run_handle.job_id,
                            self.run_handle.stage_index,
                            self.run_handle.worker_id,
                            status,
                        )
                    except Exception as e:  # noqa: BLE001
                        obs_logger.warning(
                            "lifecycle reporting failed: {}: {}",
                            type(e).__name__,
                            e,
                        )

                if execution_error is None and run_exception is None:
                    user_metrics_emitter.shutdown()
                else:
                    try:
                        user_metrics_emitter.shutdown()
                    except Exception as e:  # noqa: BLE001
                        obs_logger.warning(
                            "telemetry shutdown failed during worker failure handling: {}: {}",
                            type(e).__name__,
                            e,
                        )


__all__ = ["Worker", "WorkerRunStats", "RuntimeLifecycle"]
