from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import cloudpickle

from refiner.launchers.base import BaseLauncher
from refiner.pipeline.planning import PlannedStage
from refiner.platform.auth import MacrodataCredentialsError, current_api_key
from refiner.platform.client.api import MacrodataClient, request_json
from refiner.worker.context import logger
from refiner.worker.lifecycle import read_finalized_workers
from refiner.worker.resources.cpu import available_cpu_ids
from refiner.worker.resources.gpu import build_gpu_sets
from refiner.worker.workdir import resolve_workdir

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
        rundir: str | None = None,
        gpus_per_worker: int | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            name=name,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
        )
        self.job_id: str | None = None
        self.rundir: str | None = rundir

    @staticmethod
    def _failed_worker_payload(
        *,
        worker_id: str,
        stdout: str,
        stderr: str,
        returncode: int | None,
    ) -> dict[str, object]:
        return {
            "worker_id": worker_id,
            "claimed": 0,
            "completed": 0,
            "failed": 1,
            "output_rows": 0,
            "error": (
                stderr.strip()
                or stdout.strip()
                or f"worker process exited with code {returncode}"
            ),
        }

    def _collect_worker_results(
        self,
        *,
        stage_workers: int,
        processes: list[tuple[str, subprocess.Popen[str]]],
    ) -> LaunchStats:
        if self.job_id is None:
            raise RuntimeError("local launcher job_id is unset")
        errors: list[str] = []
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        for worker_id, process in processes:
            stdout, stderr = process.communicate()
            final_stdout_line = next(
                (line for line in reversed(stdout.splitlines()) if line.strip()),
                "",
            )
            try:
                decoded = json.loads(final_stdout_line or "{}")
                raw = (
                    decoded
                    if isinstance(decoded, dict)
                    else self._failed_worker_payload(
                        worker_id=worker_id,
                        stdout=stdout,
                        stderr=stderr,
                        returncode=process.returncode,
                    )
                )
            except json.JSONDecodeError:
                raw = self._failed_worker_payload(
                    worker_id=worker_id,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
            error = raw.get("error")
            raw_claimed = raw.get("claimed", 0)
            raw_completed = raw.get("completed", 0)
            raw_failed = raw.get("failed", 0)
            raw_output_rows = raw.get("output_rows", 0)
            result_worker_id = str(raw.get("worker_id", ""))
            result_claimed = (
                int(raw_claimed) if isinstance(raw_claimed, int | float | str) else 0
            )
            result_completed = (
                int(raw_completed)
                if isinstance(raw_completed, int | float | str)
                else 0
            )
            result_failed = (
                int(raw_failed) if isinstance(raw_failed, int | float | str) else 0
            )
            result_output_rows = (
                int(raw_output_rows)
                if isinstance(raw_output_rows, int | float | str)
                else 0
            )
            if error is not None:
                errors.append(f"worker {result_worker_id}: {error}")
            if process.returncode not in (0, None):
                errors.append(f"worker process exited with code {process.returncode}")
            claimed += result_claimed
            completed += result_completed
            failed += result_failed
            output_rows += result_output_rows

        if errors:
            raise RuntimeError("; ".join(sorted(set(errors))))

        return LaunchStats(
            job_id=self.job_id,
            workers=stage_workers,
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )

    @staticmethod
    def _ordered_shards(shards: list[Shard]) -> list[Shard]:
        return sorted(
            shards,
            key=lambda shard: (
                shard.global_ordinal is None,
                shard.global_ordinal,
                shard.id,
            ),
        )

    @staticmethod
    def _assign_shards(shards: list[Shard], *, num_workers: int) -> list[list[Shard]]:
        total = len(shards)
        base = total // num_workers
        remainder = total % num_workers
        starts = [
            worker_index * base + min(worker_index, remainder)
            for worker_index in range(num_workers)
        ]
        return [
            shards[start : start + base + (1 if worker_index < remainder else 0)]
            for worker_index, start in enumerate(starts)
        ]

    @staticmethod
    def _spawn_local_worker(
        *,
        pipeline_payload: str,
        job_id: str,
        stage_index: int,
        worker_name: str,
        worker_id: str,
        rundir: str,
        gpu_ids: tuple[str, ...],
    ) -> subprocess.Popen[str]:
        cmd = [
            sys.executable,
            "-m",
            "refiner.worker.entrypoint",
            "--pipeline-payload",
            pipeline_payload,
            "--job-id",
            job_id,
            "--stage-index",
            str(stage_index),
            "--worker-name",
            worker_name,
            "--worker-id",
            worker_id,
            "--rundir",
            rundir,
        ]
        if gpu_ids:
            cmd.extend(["--gpu-ids", ",".join(gpu_ids)])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _register_tracked_job(
        self, *, stages: list[PlannedStage]
    ) -> MacrodataClient | None:
        try:
            api_key = current_api_key()
        except MacrodataCredentialsError:
            try:
                request_json(
                    method="GET",
                    path="/api/me",
                    timeout_s=2.0,
                )
            except Exception:
                pass
            logger.warning(
                "No valid Macrodata API key found. Run `macrodata login` to track local jobs."
            )
            return None

        tracking_client = MacrodataClient(api_key=api_key)
        try:
            registered_job = tracking_client.create_job(
                name=self.name,
                executor={"type": "refiner-local"},
                plan=self._compiled_plan(stages),
                manifest=self._run_manifest(),
            )
        except Exception as err:
            logger.warning(f"Failed to register local job with Macrodata: {err}")
            return None

        self.job_id = registered_job.job_id
        if self.rundir is None:
            self.rundir = str(Path(resolve_workdir()) / "runs" / self.job_id)
        logger.info(
            "Local job registered. View job:\n  "
            f"{self._job_tracking_url(client=tracking_client, job_id=self.job_id, workspace_slug=registered_job.workspace_slug)}"
        )
        return tracking_client

    def _launch_stage(
        self,
        *,
        stage: PlannedStage,
    ) -> LaunchStats:
        # Resolve worker capacity and remaining stage shards.
        stage_workers = stage.compute.num_workers
        if self.job_id is None or self.rundir is None:
            raise RuntimeError(
                "local launcher must be initialized in launch() before running stages"
            )
        available_cpus = len(available_cpu_ids())
        if stage_workers > available_cpus:
            logger.warning(
                f"stage {stage.index} requested {stage_workers} workers, but only {available_cpus} CPUs are available on this machine."
            )
        gpu_sets = (
            build_gpu_sets(
                num_workers=stage_workers,
                gpus_per_worker=self.gpus_per_worker,
            )
            if self.gpus_per_worker is not None
            else [[] for _ in range(stage_workers)]
        )
        completed_shard_ids = {
            row.shard_id
            for row in read_finalized_workers(
                rundir=self.rundir,
                stage_index=stage.index,
            )
        }
        shards = [
            shard
            for shard in self._ordered_shards(list(stage.pipeline.list_shards()))
            if shard.id not in completed_shard_ids
        ]
        if not shards:
            logger.info(
                f"stage {stage.index}: no remaining local shards in rundir {self.rundir}"
            )
            return LaunchStats(
                job_id=self.job_id,
                workers=0,
                claimed=0,
                completed=0,
                failed=0,
                output_rows=0,
            )

        # Persist the stage payload and worker assignments under the rundir.
        stage_run_dir = Path(self.rundir) / f"stage-{stage.index}"
        stage_run_dir.mkdir(parents=True, exist_ok=True)
        payload_path = stage_run_dir / "pipeline.cloudpickle"
        payload_path.write_bytes(cloudpickle.dumps(stage.pipeline))
        shard_assignments = self._assign_shards(
            shards,
            num_workers=stage_workers,
        )

        # Spawn one subprocess per worker assignment.
        worker_ids = [uuid4().hex[:12] for _ in range(stage_workers)]
        processes: list[tuple[str, subprocess.Popen[str]]] = []
        for rank in range(stage_workers):
            worker_id = worker_ids[rank]
            assignments_path = (
                stage_run_dir / "assignments" / f"worker-{worker_id}.json"
            )
            assignments_path.parent.mkdir(parents=True, exist_ok=True)
            assignments_path.write_text(
                json.dumps(
                    [shard.to_dict() for shard in shard_assignments[rank]],
                    sort_keys=True,
                )
            )
            processes.append(
                (
                    worker_id,
                    self._spawn_local_worker(
                        pipeline_payload=str(payload_path),
                        job_id=self.job_id,
                        stage_index=stage.index,
                        worker_name=f"stage-{stage.index}-rank-{rank}",
                        worker_id=worker_id,
                        rundir=self.rundir,
                        gpu_ids=tuple(gpu_sets[rank]),
                    ),
                )
            )

        return self._collect_worker_results(
            stage_workers=stage_workers,
            processes=processes,
        )

    def launch(self) -> LaunchStats:
        available_cpus = len(available_cpu_ids())
        if self.num_workers > available_cpus:
            logger.warning(
                f"launch requested {self.num_workers} workers, but only {available_cpus} CPUs are available on this machine."
            )
        stages = self._planned_stages()
        self.job_id: str = self._build_local_job_id(self.name)
        tracking_client = self._register_tracked_job(stages=stages)
        if self.rundir is None:
            self.rundir = str(Path(resolve_workdir()) / "runs" / self.job_id)
        else:
            self.rundir = str(Path(self.rundir).expanduser().resolve())
        if self.job_id is None or self.rundir is None:
            raise RuntimeError("local launcher did not initialize job state")
        logger.info(f"Starting local job {self.job_id} with rundir={self.rundir}")
        totals = LaunchStats(
            job_id=self.job_id,
            workers=0,
            claimed=0,
            completed=0,
            failed=0,
            output_rows=0,
        )
        for stage in stages:
            if tracking_client is not None:
                try:
                    tracking_client.report_stage_started(
                        job_id=self.job_id,
                        stage_index=stage.index,
                    )
                except Exception as err:
                    logger.warning(
                        f"Failed to report local stage {stage.index} start to Macrodata: {err}"
                    )
            try:
                stage_stats = self._launch_stage(
                    stage=stage,
                )
            except Exception:
                if tracking_client is not None:
                    try:
                        tracking_client.report_stage_finished(
                            job_id=self.job_id,
                            stage_index=stage.index,
                            status="failed",
                        )
                    except Exception as err:
                        logger.warning(
                            f"Failed to report local stage {stage.index} finish to Macrodata: {err}"
                        )
                raise
            if tracking_client is not None:
                try:
                    tracking_client.report_stage_finished(
                        job_id=self.job_id,
                        stage_index=stage.index,
                        status="completed" if stage_stats.failed == 0 else "failed",
                    )
                except Exception as err:
                    logger.warning(
                        f"Failed to report local stage {stage.index} finish to Macrodata: {err}"
                    )
            totals = LaunchStats(
                job_id=self.job_id,
                workers=totals.workers + stage_stats.workers,
                claimed=totals.claimed + stage_stats.claimed,
                completed=totals.completed + stage_stats.completed,
                failed=totals.failed + stage_stats.failed,
                output_rows=totals.output_rows + stage_stats.output_rows,
            )
            if stage_stats.failed > 0:
                raise RuntimeError(
                    f"stage {stage.index} failed with {stage_stats.failed} failed shard(s)"
                )
        return totals


__all__ = ["LocalLauncher", "LaunchStats"]
