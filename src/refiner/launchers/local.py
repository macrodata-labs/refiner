from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import cloudpickle

from refiner.cli.local_run import (
    LaunchStats,
    collect_local_stage_results,
    format_resume_message,
    LocalLaunchInterrupted,
    LocalLaunchResumeError,
    stdout_is_interactive,
)
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


class LocalLauncher(BaseLauncher):
    _STAGE_HEARTBEAT_INTERVAL_SECONDS = 15.0
    _STAGE_HEARTBEAT_FAILURE_THRESHOLD = 3
    _HEARTBEAT_THREAD_JOIN_TIMEOUT_SECONDS = 5.0
    _WORKER_TERMINATE_TIMEOUT_SECONDS = 3.0

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
        self.rundir: str | None = (
            str(Path(rundir).expanduser().resolve()) if rundir is not None else None
        )
        self.job_tracking_url: str | None = None
        self._total_stages = 1

    def _collect_worker_results(
        self,
        *,
        stage_index: int,
        rundir: str,
        stage_workers: int,
        processes: list[tuple[str, subprocess.Popen[str]]],
    ) -> LaunchStats:
        if self.job_id is None:
            raise RuntimeError("local launcher job_id is unset")
        stats = collect_local_stage_results(
            job_id=self.job_id,
            job_name=self.name,
            rundir=rundir,
            stage_index=stage_index,
            total_stages=self._total_stages,
            stage_workers=stage_workers,
            tracking_url=self.job_tracking_url,
            processes=processes,
            log_mode=None,
            interrupt_message=format_resume_message(
                "Local job interrupted",
                rundir=self.rundir,
            ),
            terminate_timeout_seconds=self._WORKER_TERMINATE_TIMEOUT_SECONDS,
        )
        return stats

    def _resume_failure(self, error: str | BaseException) -> RuntimeError:
        if isinstance(error, LocalLaunchResumeError):
            return error
        return LocalLaunchResumeError(
            format_resume_message(str(error), rundir=self.rundir)
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
    ) -> tuple[MacrodataClient | None, str | None]:
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
            return None, None
        except Exception as err:
            logger.warning(
                f"Failed to load Macrodata credentials for local tracking: {err}"
            )
            return None, None

        tracking_client = MacrodataClient(api_key=api_key)
        try:
            manifest = self._run_manifest()
            manifest_environment = cast(
                dict[str, Any], manifest.setdefault("environment", {})
            )
            if not isinstance(manifest_environment, dict):
                raise RuntimeError("run manifest environment must be a dict")
            manifest_environment["rundir"] = (
                self.rundir
                if self.rundir is not None
                else str(Path(resolve_workdir()) / "runs" / "<jobid>")
            )
            registered_job = tracking_client.create_job(
                name=self.name,
                executor={"type": "refiner-local"},
                plan=self._compiled_plan(stages),
                manifest=manifest,
            )
        except MacrodataCredentialsError:
            logger.warning(
                "Your Macrodata API key is invalid. Run `macrodata login` "
                "or set MACRODATA_API_KEY with a valid key. Local execution will continue without job tracking."
            )
            return None, None
        except Exception as err:
            logger.warning(f"Failed to register local job with Macrodata: {err}")
            return None, None
        job_tracking_url = self._job_tracking_url(
            client=tracking_client,
            job_id=registered_job.job_id,
            workspace_slug=registered_job.workspace_slug,
        )
        logger.info("Local job registered. View job:\n  {}", job_tracking_url)
        self.job_tracking_url = job_tracking_url
        return tracking_client, registered_job.job_id

    def _start_stage_heartbeat(
        self,
        *,
        tracking_client: MacrodataClient,
        stage_index: int,
    ) -> tuple[threading.Event, threading.Thread]:
        if self.job_id is None:
            raise RuntimeError("local launcher job_id is unset")
        job_id = self.job_id

        stop_event = threading.Event()
        consecutive_failures = 0

        def _run() -> None:
            nonlocal consecutive_failures
            while not stop_event.wait(self._STAGE_HEARTBEAT_INTERVAL_SECONDS):
                try:
                    tracking_client.report_stage_heartbeat(
                        job_id=job_id,
                        stage_index=stage_index,
                    )
                    consecutive_failures = 0
                except Exception as err:
                    consecutive_failures += 1
                    if consecutive_failures >= self._STAGE_HEARTBEAT_FAILURE_THRESHOLD:
                        logger.warning(
                            f"Stopping local stage {stage_index} heartbeats after {consecutive_failures} consecutive failure(s): {err}"
                        )
                        stop_event.set()
                        return

        thread = threading.Thread(
            target=_run,
            name=f"refiner-stage-heartbeat-{stage_index}",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

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
            stage_index=stage.index,
            rundir=self.rundir,
            stage_workers=stage_workers,
            processes=processes,
        )

    def launch(self) -> LaunchStats:
        available_cpus = len(available_cpu_ids())
        if self.num_workers > available_cpus:
            logger.warning(
                f"launch requested {self.num_workers} workers, but only {available_cpus} CPUs are available on this machine."
            )
        self.job_tracking_url = None
        stages = self._planned_stages()
        self._total_stages = max(1, len(stages))
        tracking_client, self.job_id = self._register_tracked_job(stages=stages)
        if self.job_id is None:
            self.job_id = self._build_local_job_id(self.name)
        if self.rundir is None:
            self.rundir = str(Path(resolve_workdir()) / "runs" / self.job_id)
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
            heartbeat_stop: threading.Event | None = None
            heartbeat_thread: threading.Thread | None = None
            if tracking_client is not None:
                try:
                    tracking_client.report_stage_started(
                        job_id=self.job_id,
                        stage_index=stage.index,
                    )
                    (
                        heartbeat_stop,
                        heartbeat_thread,
                    ) = self._start_stage_heartbeat(
                        tracking_client=tracking_client,
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
            except KeyboardInterrupt:
                if tracking_client is not None:
                    try:
                        tracking_client.report_stage_finished(
                            job_id=self.job_id,
                            stage_index=stage.index,
                            status="failed",
                            reason="Local launcher interrupted",
                        )
                    except Exception as err:
                        logger.warning(
                            f"Failed to report local stage {stage.index} finish to Macrodata: {err}"
                        )
                interrupt_error = LocalLaunchInterrupted(
                    format_resume_message(
                        "Local job interrupted",
                        rundir=self.rundir,
                    )
                )
                if not stdout_is_interactive():
                    logger.warning(
                        "Local job interrupted during stage {}.",
                        stage.index,
                    )
                    logger.warning("{}", interrupt_error)
                raise interrupt_error
            except Exception as err:
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
                if isinstance(err, BrokenPipeError):
                    raise
                raise self._resume_failure(err) from err
            finally:
                if heartbeat_stop is not None:
                    heartbeat_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(
                        timeout=self._HEARTBEAT_THREAD_JOIN_TIMEOUT_SECONDS
                    )
                    if heartbeat_thread.is_alive():
                        logger.warning(
                            f"Local stage {stage.index} heartbeat thread did not stop within {self._HEARTBEAT_THREAD_JOIN_TIMEOUT_SECONDS:.1f}s"
                        )
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
                raise self._resume_failure(
                    f"stage {stage.index} failed with {stage_stats.failed} failed shard(s)",
                )
        if not stdout_is_interactive():
            logger.success(
                "Local job completed job_id={} workers={} claimed={} completed={} failed={} output_rows={} rundir={}",
                self.job_id,
                totals.workers,
                totals.claimed,
                totals.completed,
                totals.failed,
                totals.output_rows,
                self.rundir,
            )
        return totals


__all__ = ["LocalLauncher", "LaunchStats"]
