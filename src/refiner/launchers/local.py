from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle

from refiner.worker.context import RunHandle
from refiner.pipeline.planning import PlannedStage
from refiner.worker.lifecycle import LocalRuntimeLifecycle
from refiner.worker.resources.cpu import build_cpu_sets
from refiner.worker.workdir import resolve_workdir

from refiner.launchers.base import BaseLauncher

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline
    from refiner.pipeline.data.shard import Shard


@dataclass(frozen=True, slots=True)
class LaunchStats:
    job_id: str
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
        heartbeat_interval_seconds: int = 30,
        cpus_per_worker: int | None = None,
        mem_mb_per_worker: int | None = None,
        runtime_backend: str = "auto",
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            cpus_per_worker=cpus_per_worker,
            mem_mb_per_worker=mem_mb_per_worker,
        )
        if runtime_backend not in {"auto", "platform", "file"}:
            raise ValueError("runtime_backend must be one of: auto, platform, file")
        self.workdir = resolve_workdir(workdir)
        self.runtime_backend = runtime_backend

    def _launcher_run_dir(self) -> Path:
        return Path(self.workdir) / "runs" / self.job_id / "launcher"

    def _stage_run_dir(self, stage_index: int) -> Path:
        return self._launcher_run_dir() / f"stage-{stage_index}"

    def _pipeline_payload_path(self, stage_index: int) -> Path:
        return self._stage_run_dir(stage_index) / "pipeline.cloudpickle"

    def _stats_path(self, stage_index: int, rank: int) -> Path:
        return self._stage_run_dir(stage_index) / f"worker-{rank}.json"

    def _serialize_pipeline_payload(self, pipeline: RefinerPipeline) -> bytes:
        return cloudpickle.dumps(pipeline)

    def _write_pipeline_payload(
        self, *, stage_index: int, pipeline: RefinerPipeline
    ) -> Path:
        run_dir = self._stage_run_dir(stage_index)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self._pipeline_payload_path(stage_index)
        payload_path.write_bytes(self._serialize_pipeline_payload(pipeline))
        return payload_path

    def _seed_file_runtime_shards(
        self, *, stage_index: int, shards: list["Shard"]
    ) -> None:
        LocalRuntimeLifecycle(
            run=RunHandle(job_id=self.job_id, stage_index=stage_index),
            workdir=self.workdir,
        ).seed_shards(shards)

    def _resolve_platform_context(
        self, *, stages: list[PlannedStage]
    ) -> RunHandle | None:
        if self.runtime_backend == "file":
            return None
        return self._create_platform_run(
            plan=self._compiled_plan(stages),
            fail_open=self.runtime_backend != "platform",
        )

    def _effective_runtime_backend(self, *, platform_run: RunHandle | None) -> str:
        if self.runtime_backend != "auto":
            return self.runtime_backend
        return "platform" if platform_run is not None else "file"

    def _log_tracking_url(self, platform_run: RunHandle | None) -> None:
        if platform_run is None or platform_run.client is None:
            self._info(
                f"Local launch running without platform integration (job_id={self.job_id})"
            )
            return
        tracking_url = self._job_tracking_url(
            client=platform_run.client,
            job_id=platform_run.job_id,
            workspace_slug=platform_run.workspace_slug,
        )
        self._info(f"Track job here: {tracking_url}")

    def _worker_command(
        self,
        *,
        stage_index: int,
        rank: int,
        payload_path: Path,
        stats_path: Path,
        runtime_backend: str,
        cpu_ids: list[int] | None,
        platform_run: RunHandle | None,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "refiner.worker.entrypoint",
            "--job-id",
            self.job_id,
            "--workdir",
            self.workdir,
            "--heartbeat-interval-seconds",
            str(self.heartbeat_interval_seconds),
            "--runtime-backend",
            runtime_backend,
            "--pipeline-payload",
            str(payload_path),
            "--stats-path",
            str(stats_path),
            "--cpu-ids",
            ",".join(str(cpu_id) for cpu_id in cpu_ids or []),
        ]
        if self.mem_mb_per_worker is not None:
            command.extend(["--mem-mb-per-worker", str(self.mem_mb_per_worker)])
        command.extend(["--stage-index", str(stage_index)])
        if runtime_backend == "platform" and platform_run is not None:
            command.extend(
                [
                    "--worker-name",
                    f"stage-{stage_index}-rank-{rank}",
                ]
            )
        return command

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_root if not existing else f"{src_root}{os.pathsep}{existing}"
        )
        return env

    def _read_worker_stats(
        self,
        *,
        stage_index: int,
        rank: int,
        process: subprocess.Popen[str],
    ) -> tuple[int, int, int, int]:
        stats_path = self._stats_path(stage_index, rank)
        return_code = process.wait()
        if not stats_path.exists():
            raise RuntimeError(
                f"worker {rank}: missing stats file (exit code {return_code})"
            )

        try:
            stats_text = stats_path.read_text()
        except OSError as err:
            raise RuntimeError(f"worker {rank}: unreadable stats file ({err})") from err

        try:
            stats = json.loads(stats_text)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"worker {rank}: invalid stats file ({err})") from err

        if return_code != 0 or "error" in stats:
            message = str(stats.get("error") or f"exit code {return_code}")
            raise RuntimeError(f"worker {rank}: {message}")

        return (
            int(stats["claimed"]),
            int(stats["completed"]),
            int(stats["failed"]),
            int(stats["output_rows"]),
        )

    def _collect_worker_stats(
        self,
        *,
        stage_index: int,
        stage_workers: int,
        processes: list[subprocess.Popen[str]],
        stage_run: RunHandle | None,
    ) -> LaunchStats:
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        errors: list[str] = []

        for rank, process in enumerate(processes):
            try:
                worker_claimed, worker_completed, worker_failed, worker_output_rows = (
                    self._read_worker_stats(
                        stage_index=stage_index, rank=rank, process=process
                    )
                )
            except RuntimeError as err:
                errors.append(str(err))
                continue
            claimed += worker_claimed
            completed += worker_completed
            failed += worker_failed
            output_rows += worker_output_rows

        final_status = "failed" if errors or failed > 0 else "completed"
        self._finish_platform_stage(
            stage_run, stage_index=stage_index, status=final_status
        )
        if errors:
            raise RuntimeError("; ".join(errors))

        return LaunchStats(
            job_id=self.job_id,
            workers=stage_workers,
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )

    def _launch_stage(
        self,
        *,
        stage: PlannedStage,
        runtime_backend: str,
        platform_run: RunHandle | None,
    ) -> LaunchStats:
        cpu_sets = (
            build_cpu_sets(
                num_workers=stage.compute.num_workers,
                cpus_per_worker=self.cpus_per_worker,
            )
            if self.cpus_per_worker is not None
            else [None] * stage.compute.num_workers
        )
        shards = list(stage.pipeline.list_shards())
        stage_run = self._stage_run(platform_run, stage_index=stage.index)
        if runtime_backend == "platform":
            self._seed_platform_stage(stage_run, stage_index=stage.index, shards=shards)
        else:
            self._seed_file_runtime_shards(stage_index=stage.index, shards=shards)

        payload_path = self._write_pipeline_payload(
            stage_index=stage.index,
            pipeline=stage.pipeline,
        )
        processes = [
            subprocess.Popen(
                self._worker_command(
                    stage_index=stage.index,
                    rank=rank,
                    payload_path=payload_path,
                    stats_path=self._stats_path(stage.index, rank),
                    runtime_backend=runtime_backend,
                    cpu_ids=cpu_sets[rank],
                    platform_run=stage_run,
                ),
                env=self._worker_env(),
                text=True,
            )
            for rank in range(stage.compute.num_workers)
        ]
        return self._collect_worker_stats(
            stage_index=stage.index,
            stage_workers=stage.compute.num_workers,
            processes=processes,
            stage_run=stage_run,
        )

    def launch(self) -> LaunchStats:
        stages = self._planned_stages()
        platform_run = self._resolve_platform_context(stages=stages)
        runtime_backend = self._effective_runtime_backend(platform_run=platform_run)
        if runtime_backend == "platform" and platform_run is None:
            raise RuntimeError("runtime_backend=platform requires a platform context")

        self._log_tracking_url(platform_run)
        totals = LaunchStats(
            job_id=self.job_id,
            workers=0,
            claimed=0,
            completed=0,
            failed=0,
            output_rows=0,
        )
        try:
            for stage in stages:
                stage_stats = self._launch_stage(
                    stage=stage,
                    runtime_backend=runtime_backend,
                    platform_run=platform_run,
                )
                totals = LaunchStats(
                    job_id=self.job_id,
                    workers=totals.workers + stage_stats.workers,
                    claimed=totals.claimed + stage_stats.claimed,
                    completed=totals.completed + stage_stats.completed,
                    failed=totals.failed + stage_stats.failed,
                    output_rows=totals.output_rows + stage_stats.output_rows,
                )
        except Exception:
            self._finish_platform_job(platform_run, status="failed")
            raise

        final_status = "failed" if totals.failed > 0 else "completed"
        self._finish_platform_job(platform_run, status=final_status)
        return totals


__all__ = ["LocalLauncher", "LaunchStats"]
