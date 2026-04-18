from __future__ import annotations

from collections import deque
import json
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from typing import Any

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError, MacrodataClient
from refiner.platform.client.api import sanitize_terminal_text

_DEFAULT_LOG_WINDOW_MS = 60 * 60 * 1000
_MAX_LOG_SEARCH_LIMIT = 100
_MAX_METRICS_WORKER_IDS = 50
_FOLLOW_LOG_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_LOG_PAGE_LIMIT = 100
_DEFAULT_FOLLOW_LOG_PAGE_LIMIT = 500
_FOLLOW_LOG_DEDUPE_LIMIT = 100_000
_FOLLOW_LOG_MAX_DRAIN_POLLS = 5
_FOLLOW_LOG_MAX_RETRYABLE_ERRORS = 5
_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "canceled"})


def _client() -> MacrodataClient:
    return MacrodataClient()


def _print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _format_ts(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    timestamp_ms = value * 1000 if value < 100_000_000_000 else value
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    return sanitize_terminal_text(str(value))


def _progress_text(progress: Any) -> str:
    if not isinstance(progress, dict):
        return "-"
    done = progress.get("done")
    total = progress.get("total")
    if isinstance(done, int) and isinstance(total, int):
        return f"{done}/{total}"
    return "-"


def _executor_text(value: Any) -> str:
    if value == "cloud":
        return "cloud"
    if value == "local":
        return "local"
    return _safe_text(value)


def _print_table(rows: list[list[str]]) -> None:
    if not rows:
        return
    column_count = len(rows[0])
    widths = [
        max(len(row[i]) if i < len(row) else 0 for row in rows)
        for i in range(column_count)
    ]
    for index, row in enumerate(rows):
        padded = "  ".join(
            (row[i] if i < len(row) else "").ljust(widths[i])
            for i in range(column_count)
        )
        print(padded.rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def _render_list(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No jobs found.")
        return 0

    def started_by_text(item: dict[str, Any]) -> str:
        email = item.get("startedByEmail")
        username = item.get("startedByUsername")
        if isinstance(email, str) and email:
            if isinstance(username, str) and username:
                return _safe_text(f"{username} ({email})")
            return _safe_text(email)
        if isinstance(username, str) and username:
            return _safe_text(username)
        return "-"

    rows = [["ID", "Status", "Kind", "Started By", "Progress", "Created", "Name"]]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("status")),
                _executor_text(item.get("executorKind")),
                started_by_text(item),
                _progress_text(item.get("progress")),
                _format_ts(item.get("createdAt")),
                _safe_text(item.get("name")),
            ]
        )
    _print_table(rows)
    next_cursor = payload.get("nextCursor")
    if isinstance(next_cursor, str) and next_cursor:
        print(f"\nNext cursor: {_safe_text(next_cursor)}")
    return 0


def _render_job(payload: dict[str, Any]) -> int:
    job = payload.get("job")
    if not isinstance(job, dict):
        print("Job details unavailable.", file=sys.stderr)
        return 1

    print(f"Job: {_safe_text(job.get('name'))} ({_safe_text(job.get('id'))})")
    print(
        "Status:"
        f" {_safe_text(job.get('status'))}"
        f"  Kind: {_executor_text(job.get('executorKind'))}"
        f"  Progress: {_progress_text(job.get('progress'))}"
    )
    print(
        "Created:"
        f" {_format_ts(job.get('createdAt'))}"
        f"  Started: {_format_ts(job.get('startedAt'))}"
        f"  Ended: {_format_ts(job.get('endedAt'))}"
    )
    started_by_email = job.get("startedByEmail")
    started_by_username = job.get("startedByUsername")
    if isinstance(started_by_email, str) and started_by_email:
        if isinstance(started_by_username, str) and started_by_username:
            print(
                f"Started By: {_safe_text(f'{started_by_username} ({started_by_email})')}"
            )
        else:
            print(f"Started By: {_safe_text(started_by_email)}")
    elif isinstance(started_by_username, str) and started_by_username:
        print(f"Started By: {_safe_text(started_by_username)}")
    print(
        "Workers:"
        f" {_safe_text(job.get('runningWorkers'))}/{_safe_text(job.get('totalWorkers'))}"
        f"  Cost: {_safe_text(job.get('currentCostUsd'))}"
    )
    print(
        "Manifest:"
        f" {_safe_text(job.get('manifestAvailable'))}"
        f"  Logs: {_safe_text(job.get('logsAvailable'))}"
        f"  Metrics: {_safe_text(job.get('metricsAvailable'))}"
    )
    if isinstance(job.get("error"), str) and job["error"]:
        print(f"Error: {_safe_text(job.get('error'))}")

    stages = job.get("stages")
    if isinstance(stages, list) and stages:
        print("\nStages")
        rows = [["Idx", "Status", "Shards", "Workers", "Name"]]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            rows.append(
                [
                    _safe_text(stage.get("index")),
                    _safe_text(stage.get("status")),
                    f"{_safe_text(stage.get('shardDone'))}/{_safe_text(stage.get('shardTotal'))}",
                    (
                        f"{_safe_text(stage.get('runningWorkers'))}"
                        f"/{_safe_text(stage.get('completedWorkers'))}"
                        f"/{_safe_text(stage.get('totalWorkers'))}"
                    ),
                    _safe_text(stage.get("name")),
                ]
            )
        _print_table(rows)
        step_rows = [["Stage", "Step", "Type", "Name", "Summary"]]
        has_steps = False
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_steps = stage.get("steps")
            if not isinstance(stage_steps, list):
                continue
            for step in stage_steps:
                if not isinstance(step, dict):
                    continue
                has_steps = True
                step_rows.append(
                    [
                        _safe_text(stage.get("index")),
                        _safe_text(step.get("index")),
                        _safe_text(step.get("type")),
                        _safe_text(step.get("name")),
                        _step_summary_text(step.get("args")),
                    ]
                )
        if has_steps:
            print("\nSteps")
            _print_table(step_rows)
    return 0


def _render_workers(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No workers found.")
        return 0

    rows = [["UUID", "Name", "Status", "Stage", "Running", "Done", "Started", "Host"]]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("name")),
                _safe_text(item.get("status")),
                _safe_text(item.get("stageId")),
                _safe_text(item.get("runningShardCount")),
                _safe_text(item.get("completedShardCount")),
                _format_ts(item.get("startedAt")),
                _safe_text(item.get("host")),
            ]
        )
    _print_table(rows)
    page = payload.get("page")
    if isinstance(page, dict):
        next_cursor = page.get("nextCursor")
        if isinstance(next_cursor, str) and next_cursor:
            print(f"\nNext cursor: {_safe_text(next_cursor)}")
    return 0


def _render_logs(payload: dict[str, Any]) -> int:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        print("No logs found.")
        return 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        print(_format_log_entry(entry))
    if payload.get("hasOlder") is True:
        print("\nMore logs are available before this window.")
    return 0


def _format_log_entry(entry: dict[str, Any]) -> str:
    return (
        f"{_format_ts(entry.get('ts'))} "
        f"{_safe_text(entry.get('severity')).upper():<7} "
        f"{_safe_text(entry.get('workerId'))} "
        f"{_safe_text(entry.get('sourceType'))}/{_safe_text(entry.get('sourceName'))} "
        f"{_safe_text(entry.get('line'))}"
    )


def _log_entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    raw_message_hash = entry.get("messageHash")
    message_hash = (
        str(raw_message_hash)
        if isinstance(raw_message_hash, (str, int, float)) and str(raw_message_hash)
        else ""
    )
    return (
        _safe_text(entry.get("ts")),
        _safe_text(entry.get("workerId")),
        _safe_text(entry.get("sourceType")),
        _safe_text(entry.get("sourceName")),
        message_hash or _safe_text(entry.get("line")),
    )


def _next_log_cursor(payload: dict[str, Any]) -> str | None:
    cursor = payload.get("nextCursor")
    return cursor if isinstance(cursor, str) and cursor else None


def _job_status(payload: dict[str, Any]) -> str:
    job = payload.get("job")
    if not isinstance(job, dict):
        return ""
    return _safe_text(job.get("status")).lower()


def _is_retryable_api_error(err: Exception) -> bool:
    if isinstance(err, MacrodataApiError):
        return err.status in {0, 429} or err.status >= 500
    return False


def _follow_retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


def _remember_seen_key(
    *,
    key: tuple[str, str, str, str, str],
    seen_keys: set[tuple[str, str, str, str, str]],
    seen_order: deque[tuple[str, str, str, str, str]],
) -> None:
    seen_keys.add(key)
    seen_order.append(key)
    while len(seen_order) > _FOLLOW_LOG_DEDUPE_LIMIT:
        seen_keys.discard(seen_order.popleft())


def _effective_log_limit(args: Namespace) -> int:
    if isinstance(args.limit, int):
        return args.limit
    return _DEFAULT_FOLLOW_LOG_PAGE_LIMIT if args.follow else _DEFAULT_LOG_PAGE_LIMIT


def _warn_follow_skip(*, start_ms: int, end_ms: int) -> None:
    print(
        "warning: log volume is high; skipped older backlog"
        f" from {_format_ts(start_ms)} to {_format_ts(end_ms)} to stay live",
        file=sys.stderr,
    )
    print(
        "warning: request logs for fewer workers for better visibility,"
        " or rerun with --start-ms/--end-ms for full coverage",
        file=sys.stderr,
    )


def _stream_logs(
    *,
    args: Namespace,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> int:
    client = _client()
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    seen_order: deque[tuple[str, str, str, str, str]] = deque()
    current_start_ms = start_ms
    current_end_ms = end_ms
    current_cursor: str | None = None
    full_batch_polls = 0
    log_retryable_error_count = 0
    status_retryable_error_count = 0

    while True:
        try:
            payload = client.cli_get_job_logs(
                job_id=args.job_id,
                start_ms=current_start_ms,
                end_ms=current_end_ms,
                cursor=current_cursor,
                limit=limit,
                stage_index=args.stage,
                worker_id=args.worker,
                source_type=args.source_type,
                source_name=args.source_name,
                severity=args.severity,
                search=args.search,
            )
        except MacrodataApiError as err:
            if not _is_retryable_api_error(err):
                raise
            log_retryable_error_count += 1
            if log_retryable_error_count > _FOLLOW_LOG_MAX_RETRYABLE_ERRORS:
                raise
            time.sleep(_follow_retry_delay(log_retryable_error_count))
            continue
        log_retryable_error_count = 0
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = _log_entry_key(entry)
                if key in seen_keys:
                    continue
                print(_format_log_entry(entry), flush=True)
                _remember_seen_key(
                    key=key,
                    seen_keys=seen_keys,
                    seen_order=seen_order,
                )
        current_cursor = _next_log_cursor(payload)
        if current_cursor is not None:
            full_batch_polls += 1
            if full_batch_polls < _FOLLOW_LOG_MAX_DRAIN_POLLS:
                continue
            oldest_entry = entries[0] if isinstance(entries, list) and entries else None
            skipped_end_ms = (
                int(oldest_entry.get("ts"))
                if isinstance(oldest_entry, dict)
                and isinstance(oldest_entry.get("ts"), (int, float, str))
                else current_end_ms
            )
            _warn_follow_skip(start_ms=current_start_ms, end_ms=skipped_end_ms)
            current_cursor = None
            full_batch_polls = 0
            current_start_ms = current_end_ms
            next_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            current_end_ms = max(next_end_ms, current_start_ms + 1)
            continue
        else:
            full_batch_polls = 0
        try:
            job_payload = client.cli_get_job(job_id=args.job_id)
        except MacrodataApiError as err:
            if not _is_retryable_api_error(err):
                raise
            status_retryable_error_count += 1
            if status_retryable_error_count > _FOLLOW_LOG_MAX_RETRYABLE_ERRORS:
                raise
            time.sleep(_follow_retry_delay(status_retryable_error_count))
            job_payload = {}
            continue
        except MacrodataCredentialsError:
            raise
        status_retryable_error_count = 0
        if (
            current_cursor is None
            and _job_status(job_payload) in _TERMINAL_JOB_STATUSES
        ):
            return 0
        if current_cursor is None:
            current_start_ms = current_end_ms
            next_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            current_end_ms = max(next_end_ms, current_start_ms + 1)
        time.sleep(_FOLLOW_LOG_POLL_INTERVAL_SECONDS)


def _render_resource_metrics(payload: dict[str, Any]) -> int:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        print("Metrics unavailable.", file=sys.stderr)
        return 1

    resources = metrics.get("resources")
    if not isinstance(resources, list) or not resources:
        print("No resource metrics found.")
        return 0

    latest = resources[-1]
    if not isinstance(latest, dict):
        print("No resource metrics found.")
        return 0

    print(f"Job: {_safe_text(payload.get('jobId'))}")
    print(f"Stage: {_safe_text(payload.get('stageIndex'))}")
    print(f"Range: {_safe_text(payload.get('range'))}")
    print(f"Latest sample: {_format_ts(latest.get('t'))}")
    print(
        f"CPU: {_safe_text(latest.get('cpuUsage'))} / {_safe_text(latest.get('cpuQuota'))}"
        f"  Memory: {_safe_text(latest.get('memoryUsage'))} / {_safe_text(latest.get('memoryLimit'))}"
    )
    print(
        f"Network In: {_safe_text(latest.get('networkInMb'))} MB"
        f"  Network Out: {_safe_text(latest.get('networkOutMb'))} MB"
    )
    print(f"Samples: {len(resources)}")
    return 0


def _step_summary_text(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return "-"
    parts: list[str] = []
    for key in sorted(args.keys())[:3]:
        value = args.get(key)
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        elif isinstance(value, dict):
            parts.append(f"{key}={{...}}")
    return _safe_text(", ".join(parts) if parts else "{...}")


def _render_metrics(payload: dict[str, Any]) -> int:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        print("Metrics unavailable.", file=sys.stderr)
        return 1
    if not steps:
        print("No step metrics found.")
        return 0

    print(f"Job: {_safe_text(payload.get('jobId'))}")
    print(f"Stage: {_safe_text(payload.get('stageIndex'))}")
    detail_level = _safe_text(payload.get("detailLevel"))
    if detail_level != "-":
        print(f"Detail: {detail_level}")
    for step in steps:
        if not isinstance(step, dict):
            continue
        print(
            "\n"
            f"Step {_safe_text(step.get('stepIndex'))}: "
            f"{_safe_text(step.get('name'))} "
            f"({_safe_text(step.get('type'))})"
        )
        metrics = step.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            print("No metrics.")
            continue
        rows = [["Kind", "Label", "Details"]]
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            rows.append(
                [
                    _safe_text(metric.get("metricKind")),
                    _safe_text(metric.get("label")),
                    _metric_details_text(metric),
                ]
            )
        _print_table(rows)
    return 0


def _metric_details_text(metric: dict[str, Any]) -> str:
    kind = metric.get("metricKind")
    if kind == "counter":
        if "total" not in metric:
            return _safe_text(f"unit={metric.get('unit')}")
        return _safe_text(
            f"total={metric.get('total')} rate={metric.get('rateSinceStart')} per_worker={metric.get('perWorker')}"
        )
    if kind == "gauge":
        if "avgAllTime" not in metric:
            return _safe_text(
                f"kind={metric.get('kind') or '-'} unit={metric.get('unit')}"
            )
        return _safe_text(
            f"avg_all={metric.get('avgAllTime')} avg_5m={metric.get('avgLast5m')} max_5m={metric.get('maxLast5m')}"
        )
    if kind == "histogram":
        if "average" not in metric:
            return _safe_text(f"per={metric.get('per')} unit={metric.get('unit')}")
        return _safe_text(
            f"avg={metric.get('average')} total={metric.get('total')} min={metric.get('min')} max={metric.get('max')} count={metric.get('count')}"
        )
    return "-"


def _render_manifest(
    payload: dict[str, Any],
    *,
    show_runtime: bool,
    show_deps: bool,
    show_code: bool,
) -> int:
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        print("Manifest unavailable.", file=sys.stderr)
        return 1
    _ = show_runtime
    environment = manifest.get("environment")
    print("Runtime")
    if isinstance(environment, dict):
        print(f"Python: {_safe_text(environment.get('python_version'))}")
        print(f"Refiner: {_safe_text(environment.get('refiner_version'))}")
        print(f"Platform: {_safe_text(environment.get('platform'))}")
    else:
        print("-")
    if show_deps:
        dependencies = manifest.get("dependencies")
        print("\nDependencies")
        if isinstance(dependencies, list) and dependencies:
            for dependency in dependencies:
                if isinstance(dependency, dict):
                    print(
                        f"{_safe_text(dependency.get('name'))}=={_safe_text(dependency.get('version'))}"
                    )
        else:
            print("-")
    if show_code:
        script = manifest.get("script")
        print("\nCode")
        if isinstance(script, dict):
            print(f"Path: {_safe_text(script.get('path'))}")
            print(f"SHA256: {_safe_text(script.get('sha256'))}")
            if isinstance(script.get("text"), str) and script["text"]:
                safe_script_text = "".join(
                    ch
                    for ch in script["text"]
                    if ch in "\n\t"
                    or (
                        ord(ch) >= 0x20
                        and ch != "\x7f"
                        and not (0x80 <= ord(ch) <= 0x9F)
                    )
                )
                print("\n" + safe_script_text)
        else:
            print("-")
    return 0


def _render_cancel(payload: dict[str, Any]) -> int:
    job_id = payload.get("jobId", payload.get("job_id"))
    requested = payload.get("requestedOperations", payload.get("requested_operations"))
    canceled = payload.get("canceledOperations", payload.get("canceled_operations"))
    failed = payload.get("failedOperations", payload.get("failed_operations"))
    print(
        "Canceled:"
        f" {_safe_text(job_id)}"
        f"  Requested: {_safe_text(requested)}"
        f"  Canceled: {_safe_text(canceled)}"
        f"  Failed: {_safe_text(failed)}"
    )
    return 0


def _handle_error(err: Exception) -> int:
    print(_safe_text(str(err)), file=sys.stderr)
    return 1


def cmd_jobs_list(args: Namespace) -> int:
    try:
        payload = _client().cli_list_jobs(
            status=args.status,
            executor_kind=args.kind,
            me=args.me,
            limit=args.limit,
            cursor=args.cursor,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_list(payload)


def cmd_jobs_get(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_job(payload)


def cmd_jobs_manifest(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_manifest(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return (
        _print_json(payload)
        if args.json
        else _render_manifest(
            payload,
            show_runtime=args.show_runtime,
            show_deps=args.show_deps,
            show_code=args.show_code,
        )
    )


def cmd_jobs_workers(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_workers(
            job_id=args.job_id,
            stage_index=args.stage,
            limit=args.limit,
            cursor=args.cursor,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_workers(payload)


def cmd_jobs_logs(args: Namespace) -> int:
    if args.follow and args.json:
        print("--follow cannot be combined with --json.", file=sys.stderr)
        return 1
    if args.follow and args.cursor:
        print("--follow cannot be combined with --cursor.", file=sys.stderr)
        return 1
    if args.follow and args.search:
        print("--follow cannot be combined with --search.", file=sys.stderr)
        return 1
    if args.search:
        if args.stage is None:
            print("--search requires --stage.", file=sys.stderr)
            return 1
        if args.start_ms is None or args.end_ms is None:
            print(
                "--search requires explicit --start-ms and --end-ms.",
                file=sys.stderr,
            )
            return 1
        limit = _effective_log_limit(args)
        if limit > _MAX_LOG_SEARCH_LIMIT:
            print(
                f"--search supports at most {_MAX_LOG_SEARCH_LIMIT} results.",
                file=sys.stderr,
            )
            return 1
        start_ms = args.start_ms
        end_ms = args.end_ms
    else:
        limit = _effective_log_limit(args)
        end_ms = (
            args.end_ms
            if args.end_ms is not None
            else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        )
        start_ms = (
            args.start_ms
            if args.start_ms is not None
            else end_ms - _DEFAULT_LOG_WINDOW_MS
        )
    if args.follow:
        try:
            return _stream_logs(
                args=args, start_ms=start_ms, end_ms=end_ms, limit=limit
            )
        except KeyboardInterrupt:
            print("Stopped following logs.", file=sys.stderr)
            return 130
        except (MacrodataApiError, MacrodataCredentialsError) as err:
            return _handle_error(err)
    try:
        payload = _client().cli_get_job_logs(
            job_id=args.job_id,
            start_ms=start_ms,
            end_ms=end_ms,
            cursor=args.cursor,
            limit=limit,
            stage_index=args.stage,
            worker_id=args.worker,
            source_type=args.source_type,
            source_name=args.source_name,
            severity=args.severity,
            search=args.search,
        )
    except KeyboardInterrupt:
        return 130
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_logs(payload)


def cmd_jobs_metrics(args: Namespace) -> int:
    metric_labels = list(dict.fromkeys(args.metric))
    if metric_labels and args.step is None:
        print("--metric requires --step.", file=sys.stderr)
        return 1
    try:
        payload = _client().cli_get_job_step_metrics(
            job_id=args.job_id,
            stage_index=args.stage_index,
            step_index=args.step,
            metric_labels=metric_labels,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_metrics(payload)


def cmd_jobs_resource_metrics(args: Namespace) -> int:
    worker_ids = list(dict.fromkeys(args.worker_id))
    if len(worker_ids) > _MAX_METRICS_WORKER_IDS:
        print(
            f"Too many --worker-id values; maximum is {_MAX_METRICS_WORKER_IDS}.",
            file=sys.stderr,
        )
        return 1
    try:
        payload = _client().cli_get_job_metrics(
            job_id=args.job_id,
            range_value=args.range,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            bucket_count=args.bucket_count,
            stage_index=args.stage_index,
            worker_ids=worker_ids,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_resource_metrics(payload)


def cmd_jobs_cancel(args: Namespace) -> int:
    try:
        payload = _client().cli_cancel_job(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_cancel(payload)
