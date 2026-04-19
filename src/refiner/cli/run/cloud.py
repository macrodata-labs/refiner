from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from refiner.cli.run.modes import (
    CloudAttachContext,
    CloudAttachDetached,
    emit_cloud_followup_commands,
)
from refiner.cli.jobs.follow import (
    FollowLogPoller,
    TERMINAL_JOB_STATUSES as _TERMINAL_JOB_STATUSES,
    call_with_retry,
    follow_skip_message,
    format_ts as _format_ts,
    job_status as _job_status,
    safe_text as _safe_text,
)
from refiner.cli.ui.console import (
    LocalStageConsole,
    LocalStageSnapshot,
    resolve_log_mode,
    should_emit_worker_line,
)
from refiner.cli.ui import terminal as ui
from refiner.job_urls import build_job_tracking_url
from refiner.platform.client import MacrodataClient

_DEFAULT_LOG_WINDOW_MS = 60 * 60 * 1000
_DEFAULT_ATTACH_LOG_LIMIT = 500
_ATTACH_SUMMARY_INTERVAL_SECONDS = 2.0
_ATTACH_LOGS_INTERVAL_SECONDS = 1.0
_ATTACH_MAX_LOGGED_WORKERS = 4
_ATTACH_DEDUPE_LIMIT = 100_000
_ATTACH_MAX_DRAIN_POLLS = 5
_ATTACH_DRAIN_POLL_DELAY_SECONDS = 0.1
_ATTACH_MAX_RETRYABLE_ERRORS = 5
_STAGE_NOT_STARTED_STATUSES = {"queued", "pending"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _warn_follow_skip(
    *,
    context: CloudAttachContext,
    start_ms: int,
    end_ms: int,
    console: LocalStageConsole,
) -> None:
    console.emit_system(follow_skip_message(start_ms=start_ms, end_ms=end_ms))
    if context.stage_index is None:
        console.emit_system(
            f"request logs for fewer workers for better visibility, or rerun with "
            f"`macrodata jobs logs {context.job_id} --start-ms <ms> --end-ms <ms>` for full coverage"
        )
    else:
        console.emit_system(
            f"request logs for fewer workers for better visibility, or rerun with "
            f"`macrodata jobs logs {context.job_id} --stage {context.stage_index} --start-ms <ms> --end-ms <ms>` for full coverage"
        )


def _log_worker_label(entry: dict[str, Any]) -> str:
    worker_id = entry.get("workerId")
    if isinstance(worker_id, str) and worker_id.strip():
        return worker_id.strip()
    source_name = entry.get("sourceName")
    if isinstance(source_name, str) and source_name.strip():
        return source_name.strip()
    return "cloud"


def _format_attach_log_line(entry: dict[str, Any]) -> str:
    return (
        f"{_format_ts(entry.get('ts'))} "
        f"{_safe_text(entry.get('severity')).upper():<7} "
        f"{_safe_text(entry.get('sourceType'))}/{_safe_text(entry.get('sourceName'))} "
        f"{_safe_text(entry.get('line'))}"
    )


def _emit_attach_entries(
    *,
    console: LocalStageConsole,
    log_mode: str,
    selected_worker_ids: tuple[str, ...],
) -> tuple[Callable[[dict[str, Any]], None], Callable[[], tuple[str, ...]]]:
    current_selected_worker_ids = selected_worker_ids

    def _emit_entry(entry: dict[str, Any]) -> None:
        nonlocal current_selected_worker_ids
        worker_id = _safe_text(entry.get("workerId"))
        selected_worker_id = (
            current_selected_worker_ids[0] if current_selected_worker_ids else None
        )
        should_emit = should_emit_worker_line(
            log_mode=log_mode,
            worker_id=worker_id,
            selected_worker_id=selected_worker_id,
            line=_safe_text(entry.get("line")),
            severity=entry.get("severity"),
        )
        if log_mode == "one" and worker_id != "-" and not current_selected_worker_ids:
            current_selected_worker_ids = (worker_id,)
            should_emit = should_emit_worker_line(
                log_mode=log_mode,
                worker_id=worker_id,
                selected_worker_id=worker_id,
                line=_safe_text(entry.get("line")),
                severity=entry.get("severity"),
            )
        elif (
            log_mode in {"all", "errors"}
            and should_emit
            and worker_id != "-"
            and worker_id not in current_selected_worker_ids
        ):
            if len(current_selected_worker_ids) >= _ATTACH_MAX_LOGGED_WORKERS:
                return
            current_selected_worker_ids = (*current_selected_worker_ids, worker_id)
        if not should_emit:
            return
        console.emit_lines(
            worker_id=_log_worker_label(entry),
            lines=[_format_attach_log_line(entry)],
        )

    return _emit_entry, lambda: current_selected_worker_ids


def _emit_cloud_log_mode_banner(
    *,
    console: LocalStageConsole,
    log_mode: str,
    running_workers: int,
) -> None:
    if log_mode == "none":
        console.emit_system("live log lines are hidden; updating header only")
        return
    if log_mode == "one" and running_workers > 1:
        console.emit_system("showing one running worker at a time to stay live")
        return
    if log_mode in {"all", "errors"} and running_workers > _ATTACH_MAX_LOGGED_WORKERS:
        if log_mode == "errors":
            console.emit_system(
                f"showing error logs from up to {_ATTACH_MAX_LOGGED_WORKERS} of {running_workers} running workers to stay live"
            )
            return
        console.emit_system(
            f"showing up to {_ATTACH_MAX_LOGGED_WORKERS} of {running_workers} running workers to stay live"
        )


def _active_stage(job: dict[str, Any]) -> tuple[int, int]:
    stages = job.get("stages")
    if not isinstance(stages, list) or not stages:
        return 0, 1
    job_status = _safe_text(job.get("status")).lower()
    if job_status in _TERMINAL_JOB_STATUSES and job_status != "completed":
        for index in range(len(stages) - 1, -1, -1):
            stage = stages[index]
            if not isinstance(stage, dict):
                continue
            status = _safe_text(stage.get("status")).lower()
            if status not in _STAGE_NOT_STARTED_STATUSES:
                return int(stage.get("index", index) or index), len(stages)
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        status = _safe_text(stage.get("status")).lower()
        if status not in _TERMINAL_JOB_STATUSES:
            return int(stage.get("index", 0) or 0), len(stages)
    last_stage = stages[-1]
    if isinstance(last_stage, dict):
        return int(last_stage.get("index", len(stages) - 1) or 0), len(stages)
    return len(stages) - 1, len(stages)


def _elapsed_seconds(job: dict[str, Any]) -> float:
    started_at = job.get("startedAt") or job.get("createdAt")
    if not isinstance(started_at, (int, float)):
        return 0.0
    timestamp_ms = started_at * 1000 if started_at < 100_000_000_000 else started_at
    return max(0.0, time.time() - (timestamp_ms / 1000))


def _build_snapshot(
    *,
    context: CloudAttachContext,
    job_payload: dict[str, Any],
) -> LocalStageSnapshot:
    job = job_payload.get("job")
    if not isinstance(job, dict):
        raise RuntimeError("job details unavailable")
    stage_index, total_stages = _active_stage(job)
    current_stage = None
    stages = job.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if isinstance(stage, dict) and int(stage.get("index", -1)) == stage_index:
                current_stage = stage
                break
    total_workers = int(job.get("totalWorkers", 0) or 0)
    running_workers = int(job.get("runningWorkers", 0) or 0)
    completed_workers = (
        int(current_stage.get("completedWorkers", 0) or 0)
        if isinstance(current_stage, dict)
        else 0
    )
    stage_workers = (
        int(current_stage.get("totalWorkers", total_workers) or total_workers)
        if isinstance(current_stage, dict)
        else total_workers
    )
    return LocalStageSnapshot(
        job_id=context.job_id,
        job_name=context.job_name,
        rundir=None,
        stage_index=stage_index,
        total_stages=max(1, total_stages),
        stage_workers=stage_workers,
        tracking_url=context.tracking_url,
        status=_job_status(job_payload),
        worker_total=max(total_workers, stage_workers),
        worker_running=running_workers,
        worker_completed=completed_workers,
        worker_failed=0,
        elapsed_seconds=_elapsed_seconds(job),
    )


def _cloud_context_from_job_payload(
    *,
    client: MacrodataClient,
    job_id: str,
    payload: dict[str, Any],
) -> CloudAttachContext:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise RuntimeError("job details unavailable")
    workspace_slug = job.get("workspaceSlug")
    return CloudAttachContext(
        job_id=job_id,
        job_name=_safe_text(job.get("name")),
        tracking_url=build_job_tracking_url(
            client=client,
            job_id=job_id,
            workspace_slug=workspace_slug if isinstance(workspace_slug, str) else None,
        ),
        stage_index=None,
    )


def _snapshot_and_context(
    *,
    client: MacrodataClient,
    job_id: str,
    job_payload: dict[str, Any],
    stage_index_hint: int | None,
) -> tuple[CloudAttachContext, LocalStageSnapshot]:
    context = _cloud_context_from_job_payload(
        client=client,
        job_id=job_id,
        payload=job_payload,
    )
    snapshot = _build_snapshot(
        context=context,
        job_payload=job_payload,
    )
    context = replace(
        context,
        stage_index=stage_index_hint
        if stage_index_hint is not None
        else snapshot.stage_index,
    )
    return context, snapshot


def attach_to_cloud_job(
    *,
    client: MacrodataClient,
    job_id: str,
    initial_job_payload: dict[str, Any] | None = None,
    stage_index_hint: int | None = None,
    force_attach: bool = True,
) -> int:
    job_payload = initial_job_payload or client.cli_get_job(job_id=job_id)
    context, snapshot = _snapshot_and_context(
        client=client,
        job_id=job_id,
        job_payload=job_payload,
        stage_index_hint=stage_index_hint,
    )
    if not force_attach and not ui.stdout_is_interactive():
        emit_cloud_followup_commands(context=context)
        return 0
    try:
        log_mode = resolve_log_mode(None)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    console = LocalStageConsole(
        job_id=context.job_id,
        job_name=context.job_name,
        rundir=None,
        stage_index=snapshot.stage_index,
        total_stages=snapshot.total_stages,
        stage_workers=snapshot.stage_workers,
        tracking_url=context.tracking_url,
    )
    console.emit_system(f"attached to cloud job {context.job_id}")
    selected_worker_ids: tuple[str, ...] = ()
    current_stage_index = snapshot.stage_index
    _emit_cloud_log_mode_banner(
        console=console,
        log_mode=log_mode,
        running_workers=snapshot.worker_running,
    )
    closed = False
    try:
        console.apply_snapshot(snapshot)
        poller = FollowLogPoller(
            start_ms=max(0, _now_ms() - _DEFAULT_LOG_WINDOW_MS),
            end_ms=max(max(0, _now_ms() - _DEFAULT_LOG_WINDOW_MS) + 1, _now_ms()),
            dedupe_limit=_ATTACH_DEDUPE_LIMIT,
            max_drain_polls=_ATTACH_MAX_DRAIN_POLLS,
            max_retryable_errors=_ATTACH_MAX_RETRYABLE_ERRORS,
        )
        status_retryable_error_count = 0
        next_summary_at = time.monotonic() + _ATTACH_SUMMARY_INTERVAL_SECONDS
        next_logs_at = 0.0
        terminal_seen = False

        while True:
            now = time.monotonic()
            if now >= next_summary_at:
                job_payload, status_retryable_error_count = call_with_retry(
                    lambda: client.cli_get_job(job_id=job_id),
                    retryable_error_count=status_retryable_error_count,
                    max_retryable_errors=_ATTACH_MAX_RETRYABLE_ERRORS,
                )
                status_retryable_error_count = 0
                refreshed_now = time.monotonic()
                context, snapshot = _snapshot_and_context(
                    client=client,
                    job_id=job_id,
                    job_payload=job_payload,
                    stage_index_hint=stage_index_hint,
                )
                stage_changed = snapshot.stage_index != current_stage_index
                if stage_changed:
                    selected_worker_ids = ()
                    poller.reset_window(
                        now_ms=_now_ms(),
                        window_ms=_DEFAULT_LOG_WINDOW_MS,
                    )
                    next_logs_at = now
                console.apply_snapshot(snapshot)
                current_stage_index = snapshot.stage_index
                if context.stage_index != current_stage_index:
                    context = replace(context, stage_index=current_stage_index)
                if stage_changed:
                    _emit_cloud_log_mode_banner(
                        console=console,
                        log_mode=log_mode,
                        running_workers=snapshot.worker_running,
                    )
                terminal_seen = _job_status(job_payload) in _TERMINAL_JOB_STATUSES
                next_summary_at = refreshed_now + _ATTACH_SUMMARY_INTERVAL_SECONDS

            logs_available = isinstance(job_payload.get("job"), dict) and bool(
                cast(dict[str, Any], job_payload["job"]).get("logsAvailable", True)
            )
            if log_mode == "none":
                logs_available = False
            if logs_available and now >= next_logs_at:
                emit_entry, selected_worker_ids_getter = _emit_attach_entries(
                    console=console,
                    log_mode=log_mode,
                    selected_worker_ids=selected_worker_ids,
                )
                result = poller.poll(
                    fetch_page=lambda page_start_ms, page_end_ms, page_cursor: (
                        client.cli_get_job_logs(
                            job_id=job_id,
                            start_ms=page_start_ms,
                            end_ms=page_end_ms,
                            cursor=page_cursor,
                            limit=_DEFAULT_ATTACH_LOG_LIMIT,
                            stage_index=current_stage_index,
                            worker_id=None,
                            source_type=None,
                            source_name=None,
                            severity=None,
                            search=None,
                        )
                    ),
                    emit_entry=emit_entry,
                    now_ms=_now_ms,
                )
                selected_worker_ids = selected_worker_ids_getter()
                if result.action == "drain":
                    next_logs_at = time.monotonic() + _ATTACH_DRAIN_POLL_DELAY_SECONDS
                    continue
                if result.action == "skip":
                    _warn_follow_skip(
                        context=context,
                        start_ms=result.skipped_start_ms or poller.start_ms,
                        end_ms=result.skipped_end_ms or poller.end_ms,
                        console=console,
                    )
                    next_logs_at = time.monotonic()
                    continue
                next_logs_at = time.monotonic() + _ATTACH_LOGS_INTERVAL_SECONDS

            if terminal_seen and poller.cursor is None:
                console.emit_system(
                    f"cloud job {context.job_id} finished with status {_job_status(job_payload)}"
                )
                return 0

            deadline = (
                min(next_summary_at, next_logs_at)
                if logs_available
                else next_summary_at
            )
            time.sleep(max(0.05, deadline - time.monotonic()))
    except KeyboardInterrupt:
        console.emit_system(
            f"detached from cloud job {context.job_id}. The cloud job is still running."
        )
        closed = True
        console.close()
        print(f"View job: {context.tracking_url}", file=sys.stderr)
        print(f"Reattach: macrodata jobs attach {context.job_id}", file=sys.stderr)
        print(f"Cancel: macrodata jobs cancel {context.job_id}", file=sys.stderr)
        raise CloudAttachDetached()
    finally:
        if not closed:
            console.close()
