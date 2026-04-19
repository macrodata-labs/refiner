from __future__ import annotations

from collections import deque
from argparse import Namespace
from datetime import datetime, timezone
import sys
import time
from typing import Any

from refiner.cli.job_utils import (
    TERMINAL_JOB_STATUSES,
    format_ts as _format_ts,
    is_retryable_api_error as _is_retryable_api_error,
    job_status as _job_status,
    log_entry_key as _log_entry_key,
    next_log_cursor as _next_log_cursor,
    parse_epoch_ms,
    remember_seen_key as _remember_seen_key,
    safe_text as _safe_text,
)
from refiner.cli.jobs.common import _client, _handle_error, _print_json
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError

_DEFAULT_LOG_WINDOW_MS = 60 * 60 * 1000
_MAX_LOG_SEARCH_LIMIT = 100
_FOLLOW_LOG_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_LOG_PAGE_LIMIT = 100
_DEFAULT_FOLLOW_LOG_PAGE_LIMIT = 500
_FOLLOW_LOG_DEDUPE_LIMIT = 100_000
_FOLLOW_LOG_MAX_DRAIN_POLLS = 5
_FOLLOW_LOG_DRAIN_POLL_DELAY_SECONDS = 0.1
_FOLLOW_LOG_MAX_RETRYABLE_ERRORS = 5


def _format_log_entry(entry: dict[str, Any]) -> str:
    return (
        f"{_format_ts(entry.get('ts'))} "
        f"{_safe_text(entry.get('severity')).upper():<7} "
        f"{_safe_text(entry.get('workerId'))} "
        f"{_safe_text(entry.get('sourceType'))}/{_safe_text(entry.get('sourceName'))} "
        f"{_safe_text(entry.get('line'))}"
    )


def _follow_retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


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


def _emit_follow_entries(
    *,
    entries: Any,
    seen_keys: set[tuple[str, str, str, str, str]],
    seen_order: deque[tuple[str, str, str, str, str]],
) -> int | None:
    newest_entry_ms: int | None = None
    if not isinstance(entries, list):
        return newest_entry_ms
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
            limit=_FOLLOW_LOG_DEDUPE_LIMIT,
        )
        parsed_entry_ms = parse_epoch_ms(entry.get("ts"))
        if parsed_entry_ms is not None:
            newest_entry_ms = parsed_entry_ms
    return newest_entry_ms


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
        newest_processed_ms = _emit_follow_entries(
            entries=entries,
            seen_keys=seen_keys,
            seen_order=seen_order,
        )
        current_cursor = _next_log_cursor(payload)
        if current_cursor is not None:
            full_batch_polls += 1
            if full_batch_polls < _FOLLOW_LOG_MAX_DRAIN_POLLS:
                time.sleep(_FOLLOW_LOG_DRAIN_POLL_DELAY_SECONDS)
                continue
            skipped_start_ms = newest_processed_ms
            if skipped_start_ms is None:
                skipped_start_ms = current_start_ms
            _warn_follow_skip(start_ms=skipped_start_ms, end_ms=current_end_ms)
            current_cursor = None
            full_batch_polls = 0
            current_start_ms = current_end_ms
            next_end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            current_end_ms = max(next_end_ms, current_start_ms + 1)
            continue

        full_batch_polls = 0
        next_start_ms = current_end_ms
        next_end_ms = max(
            int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            next_start_ms + 1,
        )
        try:
            job_payload = client.cli_get_job(job_id=args.job_id)
        except MacrodataApiError as err:
            if not _is_retryable_api_error(err):
                raise
            status_retryable_error_count += 1
            if status_retryable_error_count > _FOLLOW_LOG_MAX_RETRYABLE_ERRORS:
                raise
            current_start_ms = next_start_ms
            current_end_ms = next_end_ms
            time.sleep(_follow_retry_delay(status_retryable_error_count))
            continue
        except MacrodataCredentialsError:
            raise
        status_retryable_error_count = 0
        if _job_status(job_payload) in TERMINAL_JOB_STATUSES:
            final_end_ms = max(
                int(datetime.now(tz=timezone.utc).timestamp() * 1000),
                next_start_ms + 1,
            )
            final_payload = client.cli_get_job_logs(
                job_id=args.job_id,
                start_ms=next_start_ms,
                end_ms=final_end_ms,
                cursor=None,
                limit=limit,
                stage_index=args.stage,
                worker_id=args.worker,
                source_type=args.source_type,
                source_name=args.source_name,
                severity=args.severity,
                search=args.search,
            )
            _emit_follow_entries(
                entries=final_payload.get("entries"),
                seen_keys=seen_keys,
                seen_order=seen_order,
            )
            return 0
        current_start_ms = next_start_ms
        current_end_ms = next_end_ms
        time.sleep(_FOLLOW_LOG_POLL_INTERVAL_SECONDS)


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
                args=args,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=limit,
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
