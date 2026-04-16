from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from typing import Any

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError, MacrodataClient
from refiner.platform.client.api import sanitize_terminal_text

_DEFAULT_LOG_WINDOW_MS = 60 * 60 * 1000


def _client() -> MacrodataClient:
    return MacrodataClient()


def _print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _format_ts(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    timestamp_ms = value * 1000 if value < 100_000_000_000 else value
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
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
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        padded = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(padded.rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _render_list(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No jobs found.")
        return 0

    rows = [["ID", "Status", "Kind", "Progress", "Created", "Name"]]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("status")),
                _executor_text(item.get("executorKind")),
                _progress_text(item.get("progress")),
                _format_ts(item.get("createdAt")),
                _safe_text(item.get("name")),
            ]
        )
    _print_table(rows)
    next_cursor = payload.get("nextCursor")
    if isinstance(next_cursor, str) and next_cursor:
        print(f"\nNext cursor: {next_cursor}")
    return 0


def _render_job(payload: dict[str, Any]) -> int:
    job = payload.get("job")
    if not isinstance(job, dict):
        print("Job details unavailable.")
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
                    f"{_safe_text(stage.get('runningWorkers'))}/{_safe_text(stage.get('totalWorkers'))}",
                    _safe_text(stage.get("name")),
                ]
            )
        _print_table(rows)
    return 0


def _render_workers(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No workers found.")
        return 0

    rows = [["Worker", "Status", "Stage", "Running", "Done", "Started", "Host"]]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("status")),
                _safe_text(item.get("stageId")),
                _safe_text(item.get("runningShardCount")),
                _safe_text(item.get("completedShardCount")),
                _format_ts(item.get("startedAt")),
                _safe_text(item.get("host")),
            ]
        )
    _print_table(rows)
    return 0


def _render_logs(payload: dict[str, Any]) -> int:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        print("No logs found.")
        return 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        line = (
            f"{_format_ts(entry.get('ts'))} "
            f"{_safe_text(entry.get('severity')).upper():<7} "
            f"{_safe_text(entry.get('workerId'))} "
            f"{_safe_text(entry.get('sourceType'))}/{_safe_text(entry.get('sourceName'))} "
            f"{_safe_text(entry.get('line'))}"
        )
        print(line)
    if payload.get("hasOlder") is True:
        print("\nMore logs are available before this window.")
    return 0


def _render_metrics(payload: dict[str, Any]) -> int:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        print("Metrics unavailable.")
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


def _render_manifest(payload: dict[str, Any]) -> int:
    print(json.dumps(payload.get("manifest"), indent=2, sort_keys=True))
    return 0


def _handle_error(err: Exception) -> int:
    print(_safe_text(str(err)))
    return 1


def cmd_jobs_list(args: Namespace) -> int:
    try:
        payload = _client().cli_list_jobs(
            status=args.status,
            executor_kind=args.kind,
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
    return _print_json(payload) if args.json else _render_manifest(payload)


def cmd_jobs_workers(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_workers(
            job_id=args.job_id,
            stage_index=args.stage,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_workers(payload)


def cmd_jobs_logs(args: Namespace) -> int:
    end_ms = args.end_ms if args.end_ms is not None else _now_ms()
    start_ms = (
        args.start_ms if args.start_ms is not None else end_ms - _DEFAULT_LOG_WINDOW_MS
    )
    try:
        payload = _client().cli_get_job_logs(
            job_id=args.job_id,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=args.limit,
            stage_index=args.stage,
            worker_id=args.worker,
            source_type=args.source_type,
            source_name=args.source_name,
            severity=args.severity,
            search=args.search,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_logs(payload)


def cmd_jobs_metrics(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_metrics(
            job_id=args.job_id,
            range_value=args.range,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            bucket_count=args.bucket_count,
            stage_index=args.stage,
            worker_ids=args.worker_id,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_metrics(payload)
