from __future__ import annotations

import json
import queue
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle

from refiner.execution.engine import block_num_rows
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.sinks import BaseSink, NullSink
from refiner.platform.client.api import MacrodataClient
from refiner.platform.client.models import FinalizedShardWorker
from refiner.worker.context import RunHandle
from refiner.worker.context import set_active_run_context, set_active_step_index
from refiner.worker.metrics.context import (
    NOOP_USER_METRICS_EMITTER,
    UserMetricsEmitter,
    set_active_user_metrics_emitter,
)
from refiner.worker.metrics.otel import OtelTelemetryEmitter
from refiner.worker.resources.cpu import set_cpu_affinity

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class LocalPlatformConfig:
    api_key: str
    base_url: str


@dataclass(frozen=True, slots=True)
class LocalWorkerConfig:
    pipeline_payload: bytes
    job_id: str
    stage_index: int
    worker_name: str
    worker_id: str
    workdir: str
    cpu_ids: list[int] | None = None
    platform: LocalPlatformConfig | None = None


@dataclass(frozen=True, slots=True)
class LocalWorkerResult:
    worker_id: str
    claimed: int
    completed: int
    failed: int
    output_rows: int
    completed_shard_ids: tuple[str, ...]
    error: str | None = None


class LocalReportingLifecycle:
    def __init__(
        self,
        *,
        run: RunHandle,
        workdir: str,
    ) -> None:
        self.run = run
        self.workdir = workdir

    def claim(self, previous: Shard | None = None) -> Shard | None:
        del previous
        raise RuntimeError("local reporting lifecycle does not support shard claims")

    def heartbeat(self, shards: list[Shard]) -> None:
        del shards
        return None

    def complete(self, shard: Shard) -> None:
        del shard
        return None

    def fail(self, shard: Shard, error: str | None = None) -> None:
        del shard, error
        return None

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        target_stage = self.run.stage_index if stage_index is None else stage_index
        if self.run.client is not None:
            response = self.run.client.shard_finalized_workers(
                job_id=self.run.job_id,
                stage_index=target_stage,
            )
            return list(response.shards)

        path = _finalized_workers_path(
            workdir=self.workdir,
            job_id=self.run.job_id,
            stage_index=target_stage,
        )
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        out: list[FinalizedShardWorker] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            shard_id = item.get("shard_id")
            worker_id = item.get("worker_id")
            if isinstance(shard_id, str) and isinstance(worker_id, str):
                out.append(FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id))
        out.sort(key=lambda row: row.shard_id)
        return out


def _finalized_workers_path(*, workdir: str, job_id: str, stage_index: int) -> Path:
    return (
        Path(workdir)
        / "runs"
        / job_id
        / "launcher"
        / "finalized-workers"
        / f"stage-{stage_index}.json"
    )


def write_finalized_workers_manifest(
    *,
    workdir: str,
    job_id: str,
    stage_index: int,
    rows: list[FinalizedShardWorker],
) -> None:
    path = _finalized_workers_path(
        workdir=workdir,
        job_id=job_id,
        stage_index=stage_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"shard_id": row.shard_id, "worker_id": row.worker_id}
        for row in sorted(rows, key=lambda row: row.shard_id)
    ]
    path.write_text(json.dumps(payload, sort_keys=True))


def _platform_client(
    platform: LocalPlatformConfig | None,
) -> MacrodataClient | None:
    if platform is None:
        return None
    return MacrodataClient(api_key=platform.api_key, base_url=platform.base_url)


def _user_metrics_emitter(
    *,
    client: MacrodataClient | None,
    job_id: str,
    stage_index: int,
    worker_id: str,
) -> UserMetricsEmitter:
    if client is None:
        return NOOP_USER_METRICS_EMITTER
    try:
        return OtelTelemetryEmitter(
            base_url=client.base_url,
            api_key=client.api_key,
            job_id=job_id,
            stage_index=stage_index,
            worker_id=worker_id,
        )
    except Exception:
        return NOOP_USER_METRICS_EMITTER


def _run_single_shard(
    *,
    pipeline: RefinerPipeline,
    sink: BaseSink,
    shard: Shard,
    sink_step_index: int | None,
) -> int:
    output_rows = 0
    for block in pipeline.execute(pipeline.source.iter_shard_units(shard)):
        with set_active_step_index(sink_step_index):
            sink.write_block(block)
        output_rows += block_num_rows(block)
    return output_rows


def run_local_worker(
    config: LocalWorkerConfig,
    task_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    result = LocalWorkerResult(
        worker_id=config.worker_id,
        claimed=0,
        completed=0,
        failed=0,
        output_rows=0,
        completed_shard_ids=(),
    )
    user_metrics_emitter: UserMetricsEmitter = NOOP_USER_METRICS_EMITTER

    try:
        if config.cpu_ids:
            set_cpu_affinity(config.cpu_ids)

        pipeline = cloudpickle.loads(config.pipeline_payload)
        client = _platform_client(config.platform)
        run_handle = RunHandle(
            job_id=config.job_id,
            stage_index=config.stage_index,
            worker_name=config.worker_name,
            worker_id=config.worker_id,
            client=client,
        )
        lifecycle = LocalReportingLifecycle(run=run_handle, workdir=config.workdir)
        user_metrics_emitter = _user_metrics_emitter(
            client=client,
            job_id=config.job_id,
            stage_index=config.stage_index,
            worker_id=config.worker_id,
        )
        sink = pipeline.sink or NullSink()
        sink_step_index = (
            pipeline._next_step_index() if pipeline.sink is not None else None
        )
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        completed_shard_ids: list[str] = []
        worker_error: str | None = None

        with (
            set_active_user_metrics_emitter(user_metrics_emitter),
            set_active_run_context(run_handle=run_handle, runtime_lifecycle=lifecycle),
        ):
            while True:
                if stop_event.is_set():
                    break
                try:
                    item = task_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break

                shard = Shard.from_dict(item)
                claimed += 1
                if client is not None:
                    client.shard_start(
                        job_id=config.job_id,
                        stage_index=config.stage_index,
                        worker_id=config.worker_id,
                        shard_id=shard.id,
                    )
                try:
                    output_rows += _run_single_shard(
                        pipeline=pipeline,
                        sink=sink,
                        shard=shard,
                        sink_step_index=sink_step_index,
                    )
                    with set_active_step_index(sink_step_index):
                        sink.on_shard_complete(shard.id)
                    user_metrics_emitter.force_flush_user_metrics()
                    if client is not None:
                        client.shard_finish(
                            job_id=config.job_id,
                            stage_index=config.stage_index,
                            worker_id=config.worker_id,
                            shard_id=shard.id,
                            status="completed",
                        )
                    completed += 1
                    completed_shard_ids.append(shard.id)
                except Exception as exc:
                    worker_error = str(exc).strip() or type(exc).__name__
                    user_metrics_emitter.force_flush_logs()
                    if client is not None:
                        client.shard_finish(
                            job_id=config.job_id,
                            stage_index=config.stage_index,
                            worker_id=config.worker_id,
                            shard_id=shard.id,
                            status="failed",
                            error=worker_error,
                        )
                    failed += 1
                    stop_event.set()
                    break

        try:
            with set_active_step_index(sink_step_index):
                sink.close()
        finally:
            user_metrics_emitter.shutdown()

        result = LocalWorkerResult(
            worker_id=config.worker_id,
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
            completed_shard_ids=tuple(completed_shard_ids),
            error=worker_error,
        )
    except Exception as exc:
        try:
            user_metrics_emitter.shutdown()
        except Exception:
            pass
        result = LocalWorkerResult(
            worker_id=config.worker_id,
            claimed=result.claimed,
            completed=result.completed,
            failed=result.failed + 1,
            output_rows=result.output_rows,
            completed_shard_ids=result.completed_shard_ids,
            error=str(exc).strip() or type(exc).__name__,
        )

    result_queue.put(asdict(result))


__all__ = [
    "LocalPlatformConfig",
    "LocalWorkerConfig",
    "LocalWorkerResult",
    "run_local_worker",
    "write_finalized_workers_manifest",
]
