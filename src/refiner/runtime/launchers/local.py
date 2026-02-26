from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cloudpickle

from refiner.ledger import FsLedger
from refiner.runtime.cpu import build_cpu_sets, set_cpu_affinity
from refiner.runtime.observer import WorkerLifecycleObserver, WorkerObserverContext
from refiner.runtime.worker import Worker, WorkerRunStats

from .base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


@dataclass(frozen=True, slots=True)
class LaunchStats:
    run_id: str
    workers: int
    claimed: int
    completed: int
    failed: int
    output_rows: int


class LocalLauncher(BaseLauncher):
    def __init__(
        self,
        *,
        pipeline: RefinerPipeline,
        name: str,
        num_workers: int = 1,
        workdir: str | None = None,
        heartbeat_every_rows: int = 4096,
        cpus_per_worker: int | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_every_rows=heartbeat_every_rows,
        )
        self.workdir = workdir
        self.cpus_per_worker = (
            int(cpus_per_worker) if cpus_per_worker is not None else None
        )
        self.ledger = FsLedger(run_id=self.run_id, worker_id=None, workdir=self.workdir)

    def _launcher_run_dir(self) -> Path:
        ledger = self._require_fs_ledger()
        return Path(ledger.workdir) / "runs" / self.run_id / "launcher"

    def _pipeline_payload_path(self) -> Path:
        return self._launcher_run_dir() / "pipeline.cloudpickle"

    def _write_pipeline_payload(self) -> Path:
        run_dir = self._launcher_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self._pipeline_payload_path()
        try:
            with payload_path.open("wb") as f:
                cloudpickle.dump(self.pipeline, f)
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

    def launch(self) -> LaunchStats:
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
            self.seed_ledger(shards=shards)
            cpu_ids = cpu_sets[0]
            old_affinity: set[int] | None = None
            if cpu_ids is not None and hasattr(os, "sched_getaffinity"):
                old_affinity = set(int(x) for x in os.sched_getaffinity(0))
                set_cpu_affinity(cpu_ids)
            ledger = FsLedger(run_id=self.run_id, worker_id=0, workdir=self.workdir)
            try:
                worker_observer = None
                if observer_ctx is not None:
                    worker_observer = WorkerLifecycleObserver(
                        client=observer_ctx.client,
                        context=WorkerObserverContext(
                            job_id=observer_ctx.job.job_id,
                            stage_id=observer_ctx.job.stage_id,
                            worker_id="local-rank-0",
                        ),
                    )
                stats = Worker(
                    rank=0,
                    ledger=ledger,
                    pipeline=self.pipeline,
                    heartbeat_every_rows=self.heartbeat_every_rows,
                    observer=worker_observer,
                ).run()
            finally:
                if old_affinity is not None:
                    os.sched_setaffinity(0, old_affinity)
            status = "failed" if stats.failed > 0 else "completed"
            self._finish_observer_terminal(observer_ctx, status=status)
            return LaunchStats(
                run_id=self.run_id,
                workers=1,
                claimed=stats.claimed,
                completed=stats.completed,
                failed=stats.failed,
                output_rows=stats.output_rows,
            )

        payload_path = self._write_pipeline_payload()
        shards = list(self.pipeline.source.list_shards())
        observer_ctx = self._setup_observer(shards=shards)
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
                "refiner.runtime.worker_entrypoint",
                "--rank",
                str(rank),
                "--run-id",
                self.run_id,
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
            if observer_ctx is not None:
                cmd.extend(
                    [
                        "--job-id",
                        observer_ctx.job.job_id,
                        "--stage-id",
                        observer_ctx.job.stage_id,
                        "--worker-id",
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
            run_id=self.run_id,
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
