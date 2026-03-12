from __future__ import annotations

from dataclasses import dataclass
import socket
import threading

from loguru import logger

from refiner.pipeline.data.shard import Shard
from refiner.platform.client import RunHandle
from refiner.pipeline.pipeline import RefinerPipeline
from refiner.execution.engine import block_num_rows
from refiner.worker.lifecycle import PlatformRuntimeLifecycle, RuntimeLifecycle
from refiner.worker.metrics.context import (
    NOOP_USER_METRICS_EMITTER,
    UserMetricsEmitter,
    set_active_step_index,
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
        rank: int,
        runtime_lifecycle: RuntimeLifecycle | None,
        pipeline: RefinerPipeline,
        *,
        heartbeat_interval_seconds: int = 30,
        run_handle: RunHandle | None = None,
    ):
        self.rank = rank
        self.runtime_lifecycle = runtime_lifecycle
        self.pipeline = pipeline
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.run_handle = run_handle

    def _start_platform_session(self) -> tuple[RuntimeLifecycle, RunHandle]:
        if self.run_handle is None or self.run_handle.client is None:
            raise ValueError("platform runtime requires a run with a client")
        try:
            host = socket.gethostname()
        except Exception:
            host = None
        started_resp = self.run_handle.client.report_worker_started(
            job_id=self.run_handle.job_id,
            stage_id=self.run_handle.stage_id,
            host=host,
            worker_name=self.run_handle.worker_name,
        )
        run = self.run_handle.with_worker(worker_id=started_resp.worker_id)
        runtime_lifecycle = PlatformRuntimeLifecycle(run=run)
        return runtime_lifecycle, run

    def run(self) -> WorkerRunStats:
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be > 0")
        if self.runtime_lifecycle is None and self.run_handle is None:
            raise ValueError(
                "runtime_lifecycle is required unless a platform run is provided"
            )

        previous: Shard | None = None
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        inflight: list[Shard] = []
        inflight_lock = threading.Lock()
        failed_error: str | None = None
        run_handle = self.run_handle
        runtime_lifecycle = self.runtime_lifecycle
        active_run: RunHandle | None = None
        user_metrics_emitter: UserMetricsEmitter = NOOP_USER_METRICS_EMITTER
        obs_logger = logger.bind(rank=self.rank)
        stop_heartbeat = threading.Event()
        heartbeat_error: Exception | None = None

        if run_handle is not None and run_handle.client is not None:
            runtime_lifecycle, active_run = self._start_platform_session()
            obs_logger.info(
                "worker started job_id={} stage_id={} worker_id={}",
                active_run.job_id,
                active_run.stage_id,
                active_run.worker_id,
            )
            try:
                telemetry_emitter = OtelTelemetryEmitter(
                    base_url=run_handle.client.base_url,
                    api_key=run_handle.client.api_key,
                    job_id=active_run.job_id,
                    stage_index=int(active_run.stage_id),
                    worker_id=active_run.worker_id or "",
                )
            except Exception as e:  # noqa: BLE001
                obs_logger.warning(
                    "telemetry setup failed: {}: {}",
                    type(e).__name__,
                    e,
                )
            else:
                user_metrics_emitter = telemetry_emitter
        if runtime_lifecycle is None:
            raise ValueError("runtime_lifecycle was not initialized")

        def _heartbeat_once() -> None:
            with inflight_lock:
                snapshot = list(inflight)
            if not snapshot:
                return
            runtime_lifecycle.heartbeat(snapshot)

        def _heartbeat_loop() -> None:
            nonlocal heartbeat_error
            while not stop_heartbeat.wait(self.heartbeat_interval_seconds):
                try:
                    _heartbeat_once()
                except Exception as e:  # noqa: BLE001
                    heartbeat_error = e
                    return

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"refiner-heartbeat-{self.rank}",
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
                    break
                claimed += 1
                with inflight_lock:
                    inflight.append(shard)
                with set_active_step_index(0):
                    yield from self.pipeline.source.iter_shard_units(shard)
                previous = shard

        with set_active_user_metrics_emitter(user_metrics_emitter):
            run_exception: Exception | None = None
            try:
                try:
                    for block in self.pipeline.execute(_source_rows()):
                        if heartbeat_error is not None:
                            raise RuntimeError(f"heartbeat failed: {heartbeat_error}")
                        produced = block_num_rows(block)
                        if produced > 0:
                            output_rows += produced
                except Exception as e:
                    failed_error = str(e)
                    with inflight_lock:
                        failed_shards = list(inflight)
                        inflight.clear()
                    for shard in failed_shards:
                        runtime_lifecycle.fail(shard, str(e))
                        user_metrics_emitter.force_flush_user_metrics()
                        failed += 1
                else:
                    _heartbeat_once()
                    with inflight_lock:
                        completed_shards = list(inflight)
                        inflight.clear()
                    for shard in completed_shards:
                        user_metrics_emitter.force_flush_user_metrics()
                        runtime_lifecycle.complete(shard)
                        completed += 1

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

                if active_run is not None and active_run.client is not None:
                    status = (
                        "failed"
                        if failed_error is not None or run_exception is not None
                        else "completed"
                    )
                    error = failed_error
                    if error is None and run_exception is not None:
                        error = str(run_exception)
                    try:
                        active_run.client.report_worker_finished(
                            job_id=active_run.job_id,
                            stage_id=active_run.stage_id,
                            worker_id=active_run.worker_id or "",
                            status=status,
                            error=error,
                        )
                        obs_logger.info(
                            "worker finished job_id={} stage_id={} worker_id={} status={}",
                            active_run.job_id,
                            active_run.stage_id,
                            active_run.worker_id,
                            status,
                        )
                    except Exception as e:  # noqa: BLE001
                        obs_logger.warning(
                            "lifecycle reporting failed: {}: {}",
                            type(e).__name__,
                            e,
                        )

                if failed_error is None and run_exception is None:
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
