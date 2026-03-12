from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from refiner.ledger import FsLedger
from refiner.runtime.planning import execution_stages_for_pipeline
from refiner.runtime.resources.cpu import build_cpu_sets, set_cpu_affinity
from refiner.runtime.resources.memory import (
    restore_memory_soft_limit,
    set_memory_soft_limit_mb,
)
from refiner.runtime.worker.runner import Worker, WorkerLifecycleContext, WorkerRunStats

from .base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class LaunchStats:
    job_id: str
    workers: int
    claimed: int
    completed: int
    failed: int
    output_rows: int


class LocalLauncher(BaseLauncher):
    """Local multi-process launcher for a pipeline.

    Args:
        pipeline: Pipeline to execute.
        name: Human-readable run name.
        num_workers: Number of local worker processes.
        workdir: Optional working directory for ledger and run artifacts.
        heartbeat_every_rows: Heartbeat cadence for worker progress reporting.
        cpus_per_worker: Optional CPU cores pinned per worker process.
        mem_mb_per_worker: Optional per-worker soft address-space limit in MB.
    """

    def __init__(
        self,
        *,
        pipeline: RefinerPipeline,
        name: str,
        num_workers: int = 1,
        workdir: str | None = None,
        heartbeat_every_rows: int = 4096,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        expand_stages: bool = True,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_every_rows=heartbeat_every_rows,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )
        self.workdir = workdir
        self.expand_stages = bool(expand_stages)
        self.ledger = FsLedger(job_id=self.job_id, worker_id=None, workdir=self.workdir)

    def _launcher_run_dir(self) -> Path:
        ledger = self._require_fs_ledger()
        return Path(ledger.workdir) / "runs" / self.job_id / "launcher"

    def _pipeline_payload_path(self) -> Path:
        return self._launcher_run_dir() / "pipeline.cloudpickle"

    def _serialize_pipeline_payload(self) -> bytes:
        try:
            return cloudpickle.dumps(self.pipeline)
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize pipeline for subprocess workers: {e}"
            ) from e

    def _write_pipeline_payload(self, payload: bytes) -> Path:
        run_dir = self._launcher_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self._pipeline_payload_path()
        try:
            with payload_path.open("wb") as f:
                f.write(payload)
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize pipeline for subprocess workers: {e}"
            ) from e
        return payload_path

    def _stats_path(self, rank: int) -> Path:
        return self._launcher_run_dir() / f"worker-{rank}.json"

    def _require_fs_ledger(self) -> FsLedger:
        if self.ledger is None:
            raise ValueError("launcher.ledger is not configured")
        return cast(FsLedger, self.ledger)

    def _reset_ledger(self) -> None:
        self.ledger = FsLedger(job_id=self.job_id, worker_id=None, workdir=self.workdir)

    def _log_tracking_url(self, observer_ctx: Any | None) -> None:
        if observer_ctx is not None:
            self._info(
                f"Track job here: {self._job_tracking_url(client=observer_ctx.client, job_id=observer_ctx.job.job_id)}"
            )
        else:
            self._info(
                f"Local launch running without observability; no tracking URL (job_id={self.job_id})"
            )

    def launch(self) -> LaunchStats:
        stage_specs = execution_stages_for_pipeline(self.pipeline)
        if self.expand_stages and len(stage_specs) > 1:
            claimed = 0
            completed = 0
            failed = 0
            output_rows = 0
            for stage in stage_specs:
                stage_workers = (
                    int(stage.num_workers)
                    if stage.num_workers is not None
                    else self.num_workers
                )
                stage_launcher = LocalLauncher(
                    pipeline=stage.pipeline,
                    name=f"{self.name}:{stage.name}",
                    num_workers=stage_workers,
                    workdir=self.workdir,
                    heartbeat_every_rows=self.heartbeat_every_rows,
                    cpus_per_worker=self.cpus_per_worker,
                    mem_mb_per_worker=self.mem_mb_per_worker,
                    expand_stages=False,
                )
                stage_stats = stage_launcher.launch()
                claimed += stage_stats.claimed
                completed += stage_stats.completed
                failed += stage_stats.failed
                output_rows += stage_stats.output_rows
            return LaunchStats(
                job_id=self.job_id,
                workers=self.num_workers,
                claimed=claimed,
                completed=completed,
                failed=failed,
                output_rows=output_rows,
            )

        self.pipeline.prepare_sinks_for_launch()
        cpu_sets = (
            build_cpu_sets(
                num_workers=self.num_workers,
                cpus_per_worker=self.cpus_per_worker,
            )
            if self.cpus_per_worker is not None
            else [None] * self.num_workers
        )

        if self.num_workers == 1:
            shards = list(self.pipeline.source.list_shards())
            observer_ctx = self._setup_observer(shards=shards)
            self._log_tracking_url(observer_ctx)
            self._reset_ledger()
            self.seed_ledger(shards=shards)
            cpu_ids = cpu_sets[0]
            old_affinity: set[int] | None = None
            old_mem_limits: tuple[int, int] | None = None
            if cpu_ids is not None and hasattr(os, "sched_getaffinity"):
                old_affinity = set(int(x) for x in os.sched_getaffinity(0))
                set_cpu_affinity(cpu_ids)
            if self.mem_mb_per_worker is not None:
                old_mem_limits = set_memory_soft_limit_mb(self.mem_mb_per_worker)
            ledger = FsLedger(job_id=self.job_id, worker_id=0, workdir=self.workdir)
            try:
                lifecycle_client = None
                lifecycle_context = WorkerLifecycleContext(
                    job_id=self.job_id,
                    stage_id="",
                    worker_id="",
                    worker_name="local-rank-0",
                )
                if observer_ctx is not None:
                    worker_name = "local-rank-0"
                    try:
                        host = socket.gethostname()
                    except Exception:
                        host = None
                    started_resp = observer_ctx.client.report_worker_started(
                        job_id=observer_ctx.job.job_id,
                        stage_id=observer_ctx.job.stage_id,
                        host=host,
                        worker_name=worker_name,
                    )
                    reported_worker_id = started_resp.get("worker_id")
                    if isinstance(reported_worker_id, str) and reported_worker_id:
                        lifecycle_client = observer_ctx.client
                        lifecycle_context = WorkerLifecycleContext(
                            job_id=observer_ctx.job.job_id,
                            stage_id=observer_ctx.job.stage_id,
                            worker_id=reported_worker_id,
                            worker_name=worker_name,
                        )
                stats = Worker(
                    rank=0,
                    ledger=ledger,
                    pipeline=self.pipeline,
                    heartbeat_every_rows=self.heartbeat_every_rows,
                    lifecycle_client=lifecycle_client,
                    lifecycle_context=lifecycle_context,
                ).run()
            finally:
                if old_affinity is not None:
                    set_affinity = getattr(os, "sched_setaffinity", None)
                    if callable(set_affinity):
                        set_affinity(0, old_affinity)
                if old_mem_limits is not None:
                    restore_memory_soft_limit(old_mem_limits)
            status = "failed" if stats.failed > 0 else "completed"
            self._finish_observer_terminal(observer_ctx, status=status)
            return LaunchStats(
                job_id=self.job_id,
                workers=1,
                claimed=stats.claimed,
                completed=stats.completed,
                failed=stats.failed,
                output_rows=stats.output_rows,
            )

        payload = self._serialize_pipeline_payload()
        shards = list(self.pipeline.source.list_shards())
        observer_ctx = self._setup_observer(shards=shards)
        self._log_tracking_url(observer_ctx)
        self._reset_ledger()
        payload_path = self._write_pipeline_payload(payload)
        self.seed_ledger(shards=shards)
        procs: list[subprocess.Popen[str]] = []
        for rank in range(self.num_workers):
            stats_path = self._stats_path(rank)
            cpu_ids = cpu_sets[rank]
            cpu_arg = (
                ",".join(str(int(x)) for x in cpu_ids) if cpu_ids is not None else ""
            )
            cmd = [
                sys.executable,
                "-m",
                "refiner.runtime.worker.entrypoint",
                "--rank",
                str(rank),
                "--job-id",
                self.job_id,
                "--workdir",
                self._require_fs_ledger().workdir,
                "--heartbeat-every-rows",
                str(self.heartbeat_every_rows),
                "--pipeline-payload",
                str(payload_path),
                "--stats-path",
                str(stats_path),
                "--cpu-ids",
                cpu_arg,
            ]
            if self.mem_mb_per_worker is not None:
                cmd.extend(["--mem-mb-per-worker", str(self.mem_mb_per_worker)])
            if observer_ctx is not None:
                cmd.extend(
                    [
                        "--stage-id",
                        observer_ctx.job.stage_id,
                        "--worker-name",
                        f"local-rank-{rank}",
                    ]
                )
            p = subprocess.Popen(cmd, text=True)
            procs.append(p)

        errors: list[str] = []
        agg = WorkerRunStats()
        for rank, p in enumerate(procs):
            rc = p.wait()
            stats_path = self._stats_path(rank)
            if not stats_path.exists():
                errors.append(f"worker {rank}: missing stats file (exit code {rc})")
                continue
            try:
                msg = json.loads(stats_path.read_text())
            except Exception as e:
                errors.append(f"worker {rank}: invalid stats file ({e})")
                continue
            if rc != 0 or "error" in msg:
                err = str(msg.get("error") or f"exit code {rc}")
                errors.append(f"worker {rank}: {err}")
                continue
            agg = WorkerRunStats(
                claimed=agg.claimed + int(msg["claimed"]),
                completed=agg.completed + int(msg["completed"]),
                failed=agg.failed + int(msg["failed"]),
                output_rows=agg.output_rows + int(msg["output_rows"]),
            )

        final_status = "failed" if errors or agg.failed > 0 else "completed"
        self._finish_observer_terminal(observer_ctx, status=final_status)
        if errors:
            raise RuntimeError("; ".join(errors))

        return LaunchStats(
            job_id=self.job_id,
            workers=self.num_workers,
            claimed=agg.claimed,
            completed=agg.completed,
            failed=agg.failed,
            output_rows=agg.output_rows,
        )


__all__ = [
    "LocalLauncher",
    "LaunchStats",
]
