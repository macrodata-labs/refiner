from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TextIO, cast

from refiner.cli.job_utils import (
    TERMINAL_JOB_STATUSES as _TERMINAL_JOB_STATUSES,
    format_ts as _format_ts,
    is_retryable_api_error as _is_retryable_api_error,
    job_status as _job_status,
    log_entry_key as _log_entry_key,
    next_log_cursor as _next_log_cursor,
    parse_epoch_ms,
    remember_seen_key as _remember_seen_key,
    safe_text as _safe_text,
)
from refiner.cli.local_run import (
    LocalStageConsole,
    LocalStageSnapshot,
    stdout_is_interactive,
)
from refiner.job_urls import build_job_tracking_url
from refiner.platform.client import MacrodataApiError, MacrodataClient

_ATTACH_MODE_ENV_VAR = "REFINER_ATTACH"
_VALID_ATTACH_MODES = {"auto", "attach", "detach"}
_DEFAULT_LOG_WINDOW_MS = 60 * 60 * 1000
_DEFAULT_ATTACH_LOG_LIMIT = 500
_ATTACH_SUMMARY_INTERVAL_SECONDS = 2.0
_ATTACH_LOGS_INTERVAL_SECONDS = 1.0
_ATTACH_MAX_LOGGED_WORKERS = 4
_ATTACH_DEDUPE_LIMIT = 100_000
_ATTACH_MAX_DRAIN_POLLS = 5
_ATTACH_DRAIN_POLL_DELAY_SECONDS = 0.1
_ATTACH_MAX_RETRYABLE_ERRORS = 5


class CloudAttachDetached(KeyboardInterrupt):
    pass


@dataclass(frozen=True, slots=True)
class CloudAttachContext:
    job_id: str
    job_name: str
    tracking_url: str
    stage_index: int | None


def normalize_attach_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _VALID_ATTACH_MODES:
        allowed = ", ".join(sorted(_VALID_ATTACH_MODES))
        raise ValueError(
            f"unsupported attach mode {mode!r}; expected one of: {allowed}"
        )
    return normalized


def attach_mode_override() -> str | None:
    import os

    value = os.environ.get(_ATTACH_MODE_ENV_VAR)
    if value is None:
        return None
    return normalize_attach_mode(value)


def resolve_attach_mode() -> str:
    override = attach_mode_override()
    if override is not None and override != "auto":
        return override
    return "attach" if stdout_is_interactive() else "detach"


def require_cloud_attach_supported(executor_kind: str) -> None:
    override = attach_mode_override()
    if override is None:
        return
    if executor_kind != "cloud" and override == "detach":
        raise SystemExit("--detach is only supported for cloud launches.")


def emit_cloud_followup_commands(
    *,
    context: CloudAttachContext,
    file: TextIO = sys.stdout,
) -> None:
    print("Cloud job submitted.", file=file)
    print(f"Job ID: {context.job_id}", file=file)
    print(f"URL: {context.tracking_url}", file=file)
    print(f"Attach: macrodata jobs attach {context.job_id}", file=file)
    print(f"Summary: macrodata jobs get {context.job_id}", file=file)
    if context.stage_index is None:
        print(f"Logs: macrodata jobs logs {context.job_id}", file=file)
    else:
        print(
            f"Logs: macrodata jobs logs {context.job_id} --stage {context.stage_index}",
            file=file,
        )
    print(f"Workers: macrodata jobs workers {context.job_id}", file=file)
    print(f"Cancel: macrodata jobs cancel {context.job_id}", file=file)


def _retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


def _warn_follow_skip(
    *,
    context: CloudAttachContext,
    start_ms: int,
    end_ms: int,
    console: LocalStageConsole,
) -> None:
    console.emit_system(
        "log volume is high; skipped older backlog from "
        f"{_format_ts(start_ms)} to {_format_ts(end_ms)} to stay live"
    )
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


def _active_stage(job: dict[str, Any]) -> tuple[int, int]:
    stages = job.get("stages")
    if not isinstance(stages, list) or not stages:
        return 0, 1
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
            if (
                isinstance(stage, dict)
                and int(stage.get("index", -1) or -1) == stage_index
            ):
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
        rundir="cloud",
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
    stage_index_hint: int | None = None,
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
        stage_index=stage_index_hint,
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
        stage_index_hint=stage_index_hint,
    )
    snapshot = _build_snapshot(
        context=context,
        job_payload=job_payload,
    )
    return context, snapshot


def attach_to_cloud_job(
    *,
    client: MacrodataClient,
    job_id: str,
    initial_job_payload: dict[str, Any] | None = None,
    stage_index_hint: int | None = None,
) -> int:
    if not stdout_is_interactive():
        payload = initial_job_payload or client.cli_get_job(job_id=job_id)
        context = _cloud_context_from_job_payload(
            client=client,
            job_id=job_id,
            payload=payload,
            stage_index_hint=stage_index_hint,
        )
        emit_cloud_followup_commands(context=context)
        return 0

    job_payload = initial_job_payload or client.cli_get_job(job_id=job_id)
    context, snapshot = _snapshot_and_context(
        client=client,
        job_id=job_id,
        job_payload=job_payload,
        stage_index_hint=stage_index_hint,
    )
    console = LocalStageConsole(
        job_id=context.job_id,
        job_name=context.job_name,
        rundir="cloud",
        stage_index=snapshot.stage_index,
        total_stages=snapshot.total_stages,
        stage_workers=snapshot.stage_workers,
        tracking_url=context.tracking_url,
    )
    console.emit_system(f"attached to cloud job {context.job_id}")
    selected_worker_ids: tuple[str, ...] = ()
    current_stage_index = snapshot.stage_index
    if snapshot.worker_running > _ATTACH_MAX_LOGGED_WORKERS:
        console.emit_system(
            f"showing up to {_ATTACH_MAX_LOGGED_WORKERS} of {snapshot.worker_running} running workers to stay live"
        )
    closed = False
    try:
        console.apply_snapshot(snapshot)
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        current_start_ms = max(
            0,
            now_ms - _DEFAULT_LOG_WINDOW_MS,
        )
        current_end_ms = max(current_start_ms + 1, now_ms)
        current_cursor: str | None = None
        seen_keys: set[tuple[str, str, str, str, str]] = set()
        seen_order: deque[tuple[str, str, str, str, str]] = deque()
        full_batch_polls = 0
        log_retryable_error_count = 0
        status_retryable_error_count = 0
        next_summary_at = time.monotonic() + _ATTACH_SUMMARY_INTERVAL_SECONDS
        next_logs_at = 0.0
        terminal_seen = False

        while True:
            now = time.monotonic()
            if now >= next_summary_at:
                try:
                    job_payload = client.cli_get_job(job_id=job_id)
                except MacrodataApiError as err:
                    if not _is_retryable_api_error(err):
                        raise
                    status_retryable_error_count += 1
                    if status_retryable_error_count > _ATTACH_MAX_RETRYABLE_ERRORS:
                        raise
                    time.sleep(_retry_delay(status_retryable_error_count))
                    continue
                status_retryable_error_count = 0
                context, snapshot = _snapshot_and_context(
                    client=client,
                    job_id=job_id,
                    job_payload=job_payload,
                    stage_index_hint=stage_index_hint,
                )
                if snapshot.stage_index != current_stage_index:
                    selected_worker_ids = ()
                    current_cursor = None
                    full_batch_polls = 0
                    seen_keys.clear()
                    seen_order.clear()
                    reset_now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                    current_start_ms = max(0, reset_now_ms - _DEFAULT_LOG_WINDOW_MS)
                    current_end_ms = max(current_start_ms + 1, reset_now_ms)
                    next_logs_at = now
                console.apply_snapshot(snapshot)
                current_stage_index = snapshot.stage_index
                terminal_seen = _job_status(job_payload) in _TERMINAL_JOB_STATUSES
                next_summary_at = now + _ATTACH_SUMMARY_INTERVAL_SECONDS

            logs_available = isinstance(job_payload.get("job"), dict) and bool(
                cast(dict[str, Any], job_payload["job"]).get("logsAvailable", True)
            )
            if not logs_available:
                next_logs_at = now + _ATTACH_LOGS_INTERVAL_SECONDS
            if logs_available and now >= next_logs_at:
                try:
                    payload = client.cli_get_job_logs(
                        job_id=job_id,
                        start_ms=current_start_ms,
                        end_ms=current_end_ms,
                        cursor=current_cursor,
                        limit=_DEFAULT_ATTACH_LOG_LIMIT,
                        stage_index=current_stage_index,
                        worker_id=None,
                        source_type=None,
                        source_name=None,
                        severity=None,
                        search=None,
                    )
                except MacrodataApiError as err:
                    if not _is_retryable_api_error(err):
                        raise
                    log_retryable_error_count += 1
                    if log_retryable_error_count > _ATTACH_MAX_RETRYABLE_ERRORS:
                        raise
                    time.sleep(_retry_delay(log_retryable_error_count))
                    continue
                log_retryable_error_count = 0
                entries = payload.get("entries")
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        worker_id = _safe_text(entry.get("workerId"))
                        if (
                            worker_id != "-"
                            and worker_id not in selected_worker_ids
                            and len(selected_worker_ids) < _ATTACH_MAX_LOGGED_WORKERS
                        ):
                            selected_worker_ids = (*selected_worker_ids, worker_id)
                        if (
                            selected_worker_ids
                            and worker_id != "-"
                            and worker_id not in selected_worker_ids
                        ):
                            continue
                        key = _log_entry_key(entry)
                        if key in seen_keys:
                            continue
                        console.emit_lines(
                            worker_id=_log_worker_label(entry),
                            lines=[_format_attach_log_line(entry)],
                        )
                        _remember_seen_key(
                            key=key,
                            seen_keys=seen_keys,
                            seen_order=seen_order,
                            limit=_ATTACH_DEDUPE_LIMIT,
                        )
                current_cursor = _next_log_cursor(payload)
                if current_cursor is not None:
                    full_batch_polls += 1
                    if full_batch_polls < _ATTACH_MAX_DRAIN_POLLS:
                        next_logs_at = (
                            time.monotonic() + _ATTACH_DRAIN_POLL_DELAY_SECONDS
                        )
                        continue
                    oldest_entry = (
                        entries[0] if isinstance(entries, list) and entries else None
                    )
                    skipped_end_ms = current_end_ms
                    if isinstance(oldest_entry, dict):
                        parsed_oldest_ms = parse_epoch_ms(oldest_entry.get("ts"))
                        if parsed_oldest_ms is not None:
                            skipped_end_ms = parsed_oldest_ms
                    _warn_follow_skip(
                        context=context,
                        start_ms=current_start_ms,
                        end_ms=skipped_end_ms,
                        console=console,
                    )
                    current_cursor = None
                    full_batch_polls = 0
                    current_start_ms = current_end_ms
                    next_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                    current_end_ms = max(next_end_ms, current_start_ms + 1)
                    next_logs_at = time.monotonic()
                    continue
                full_batch_polls = 0
                current_start_ms = current_end_ms
                next_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                current_end_ms = max(next_end_ms, current_start_ms + 1)
                next_logs_at = now + _ATTACH_LOGS_INTERVAL_SECONDS

            if terminal_seen and current_cursor is None:
                console.emit_system(
                    f"cloud job {context.job_id} finished with status {_job_status(job_payload)}"
                )
                return 0

            deadline = min(next_summary_at, next_logs_at)
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
