from __future__ import annotations

import io
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import deque

from refiner.cli.ui.console import (
    StageSnapshot,
    resolve_log_mode,
    run_stage_ui,
)


class LocalLaunchResumeError(RuntimeError):
    pass


class LocalLaunchInterrupted(KeyboardInterrupt):
    pass


@dataclass(frozen=True, slots=True)
class LaunchStats:
    job_id: str
    workers: int
    claimed: int
    completed: int
    failed: int
    output_rows: int


@dataclass(slots=True)
class WorkerProcessMonitor:
    worker_id: str
    process: subprocess.Popen[str]
    stdout_buffer: list[str]
    stderr_buffer: list[str]
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread


@dataclass(frozen=True, slots=True)
class WorkerResult:
    worker_id: str
    claimed: int
    completed: int
    failed: int
    output_rows: int
    error: str | None
    returncode: int | None


def format_resume_message(message: str, *, rundir: str | None) -> str:
    suffix = (
        f" To resume completed shards, rerun with rundir={rundir!r}."
        if rundir is not None
        else ""
    )
    return message.rstrip(".") + "." + suffix


def stop_worker_monitors(
    monitors: list[WorkerProcessMonitor],
    *,
    terminate_timeout_seconds: float,
) -> None:
    running = [monitor for monitor in monitors if monitor.process.poll() is None]
    for monitor in running:
        try:
            monitor.process.terminate()
        except Exception:
            continue
    deadline = time.monotonic() + terminate_timeout_seconds
    for monitor in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            monitor.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                monitor.process.kill()
            except Exception:
                pass
    for monitor in monitors:
        for thread in (monitor.stdout_thread, monitor.stderr_thread):
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        try:
            monitor.process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass


def collect_local_stage_results(
    *,
    job_id: str,
    job_name: str,
    rundir: str,
    stage_index: int,
    total_stages: int,
    stage_workers: int,
    tracking_url: str | None,
    processes: list[tuple[str, subprocess.Popen[str]]],
    log_mode: str | None,
    interrupt_message: str,
    terminate_timeout_seconds: float,
) -> LaunchStats:
    console = None
    monitors: list[WorkerProcessMonitor] = []
    for worker_id, process in processes:
        stdout_buffer: list[str] = []
        stderr_buffer: list[str] = []
        stdout_thread = threading.Thread(
            target=_drain_stream,
            kwargs={"stream": process.stdout, "target": stdout_buffer},
            name=f"refiner-worker-stdout-{worker_id}",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_stream,
            kwargs={"stream": process.stderr, "target": stderr_buffer},
            name=f"refiner-worker-stderr-{worker_id}",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        monitors.append(
            WorkerProcessMonitor(
                worker_id=worker_id,
                process=process,
                stdout_buffer=stdout_buffer,
                stderr_buffer=stderr_buffer,
                stdout_thread=stdout_thread,
                stderr_thread=stderr_thread,
            )
        )
    worker_log_paths = {
        worker_id: Path(rundir)
        / f"stage-{stage_index}"
        / "logs"
        / f"worker-{worker_id}.log"
        for worker_id, _ in processes
    }
    stage_started_at = time.monotonic()

    def snapshot() -> StageSnapshot:
        completed = 0
        failed = 0
        for monitor in monitors:
            returncode = monitor.process.poll()
            if returncode is None:
                continue
            if returncode == 0:
                completed += 1
            else:
                failed += 1
        running = max(0, len(monitors) - completed - failed)
        return StageSnapshot(
            job_id=job_id,
            job_name=job_name,
            rundir=rundir,
            stage_index=stage_index,
            total_stages=total_stages,
            stage_workers=stage_workers,
            tracking_url=tracking_url,
            status="running" if running else ("failed" if failed > 0 else "completed"),
            worker_total=len(monitors),
            worker_running=running,
            worker_completed=completed,
            worker_failed=failed,
            elapsed_seconds=time.monotonic() - stage_started_at,
        )

    try:
        try:
            resolved_log_mode = resolve_log_mode(log_mode)
        except ValueError as err:
            raise SystemExit(str(err)) from err
        console, _ = run_stage_ui(
            worker_log_paths=worker_log_paths,
            snapshot_getter=snapshot,
            log_mode=resolved_log_mode,
            interrupt_message=interrupt_message,
            keep_open=True,
        )
        errors: list[str] = []
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        completed_workers = 0
        failed_workers = 0
        failed_worker_ids: list[str] = []
        for monitor in monitors:
            worker_result = _collect_worker_result(monitor)
            claimed += worker_result.claimed
            completed += worker_result.completed
            failed += worker_result.failed
            output_rows += worker_result.output_rows
            worker_errors = _worker_errors(worker_result)
            if worker_errors:
                failed_workers += 1
                failed_worker_ids.append(worker_result.worker_id)
                errors.extend(worker_errors)
            else:
                completed_workers += 1
        console.apply_snapshot(
            StageSnapshot(
                job_id=job_id,
                job_name=job_name,
                rundir=rundir,
                stage_index=stage_index,
                total_stages=total_stages,
                stage_workers=stage_workers,
                tracking_url=tracking_url,
                status="failed" if failed_workers > 0 else "completed",
                worker_total=len(monitors),
                worker_running=0,
                worker_completed=completed_workers,
                worker_failed=failed_workers,
                elapsed_seconds=time.monotonic() - stage_started_at,
            )
        )
        if errors:
            _emit_failed_worker_log_tails(
                console=console,
                worker_ids=failed_worker_ids,
                worker_log_paths=worker_log_paths,
            )
            console.close()
            console = None
            raise RuntimeError("; ".join(sorted(set(errors))))
        console.close()
        console = None
        return LaunchStats(
            job_id=job_id,
            workers=stage_workers,
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )
    except BaseException:
        stop_worker_monitors(
            monitors,
            terminate_timeout_seconds=terminate_timeout_seconds,
        )
        if console is not None:
            console.close()
        raise


def _collect_worker_result(monitor: WorkerProcessMonitor) -> WorkerResult:
    monitor.stdout_thread.join()
    monitor.stderr_thread.join()
    monitor.process.wait()
    stdout = "".join(monitor.stdout_buffer)
    stderr = "".join(monitor.stderr_buffer)
    final_stdout_line = next(
        (line for line in reversed(stdout.splitlines()) if line.strip()),
        "",
    )
    try:
        decoded = json.loads(final_stdout_line or "{}")
    except json.JSONDecodeError:
        decoded = None
    raw = (
        decoded
        if isinstance(decoded, dict)
        else {
            "worker_id": monitor.worker_id,
            "claimed": 0,
            "completed": 0,
            "failed": 1,
            "output_rows": 0,
            "error": (
                stderr.strip()
                or stdout.strip()
                or f"worker process exited with code {monitor.process.returncode}"
            ),
        }
    )
    return WorkerResult(
        worker_id=str(raw.get("worker_id", monitor.worker_id)),
        claimed=_coerce_int(raw.get("claimed", 0)),
        completed=_coerce_int(raw.get("completed", 0)),
        failed=_coerce_int(raw.get("failed", 0)),
        output_rows=_coerce_int(raw.get("output_rows", 0)),
        error=_coerce_error(raw.get("error")),
        returncode=monitor.process.returncode,
    )


def _coerce_int(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return 0


def _coerce_error(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _worker_errors(worker_result: WorkerResult) -> list[str]:
    errors: list[str] = []
    if worker_result.error is not None:
        errors.append(f"worker {worker_result.worker_id}: {worker_result.error}")
    if worker_result.returncode not in (0, None):
        errors.append(
            f"worker {worker_result.worker_id} exited with code {worker_result.returncode}"
        )
    if worker_result.failed > 0 and worker_result.error is None:
        errors.append(
            f"worker {worker_result.worker_id} reported {worker_result.failed} failed shard(s)"
        )
    return errors


def _emit_failed_worker_log_tails(
    *,
    console,
    worker_ids: list[str],
    worker_log_paths: dict[str, Path],
    max_lines_per_worker: int = 8,
) -> None:
    for worker_id in worker_ids:
        path = worker_log_paths.get(worker_id)
        if path is None or not path.exists():
            continue
        tail_lines = _tail_lines(path, max_lines=max_lines_per_worker)
        if not tail_lines:
            continue
        console.emit_system(f"last log lines from {worker_id}:")
        console.emit_lines(worker_id=worker_id, lines=tail_lines)


def _tail_lines(path: Path, *, max_lines: int) -> list[str]:
    lines: deque[str] = deque(maxlen=max_lines)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def _drain_stream(
    stream: io.TextIOWrapper | None,
    *,
    target: list[str],
) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            target.append(line)
    finally:
        stream.close()


__all__ = [
    "LocalLaunchInterrupted",
    "LocalLaunchResumeError",
    "LaunchStats",
    "StageSnapshot",
    "collect_local_stage_results",
    "format_resume_message",
]
