from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
import json
import sys
import time
from typing import Any, cast

from refiner.cli.jobs.follow import (
    FollowLogPoller,
    TERMINAL_JOB_STATUSES,
    call_with_retry,
    follow_skip_message,
    format_ts as _format_ts,
    is_retryable_api_error as _is_retryable_api_error,
    job_status as _job_status,
    next_log_cursor as _next_log_cursor,
    retry_delay,
    safe_text as _safe_text,
)
from refiner.cli.jobs.common import _client, _handle_error
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


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _format_log_entry(entry: dict[str, Any]) -> str:
    return (
        f"{_format_ts(entry.get('ts'))} "
        f"{_safe_text(entry.get('severity')).upper():<7} "
        f"{_safe_text(entry.get('workerId'))} "
        f"{_safe_text(entry.get('sourceType'))}/{_safe_text(entry.get('sourceName'))} "
        f"{_safe_text(entry.get('line'))}"
    )


def _effective_log_limit(args: Namespace) -> int:
    if isinstance(args.limit, int):
        return args.limit
    return _DEFAULT_FOLLOW_LOG_PAGE_LIMIT if args.follow else _DEFAULT_LOG_PAGE_LIMIT


def _warn_follow_skip(*, start_ms: int, end_ms: int) -> None:
    print(
        f"warning: {follow_skip_message(start_ms=start_ms, end_ms=end_ms)}",
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


def _fetch_log_page(
    *,
    client: Any,
    args: Namespace,
    start_ms: int,
    end_ms: int,
    cursor: str | None,
    limit: int,
) -> dict[str, Any]:
    return client.cli_get_job_logs(
        job_id=args.job_id,
        start_ms=start_ms,
        end_ms=end_ms,
        cursor=cursor,
        limit=limit,
        stage_index=args.stage,
        worker_id=args.worker,
        source_type=args.source_type,
        source_name=args.source_name,
        severity=args.severity,
        search=args.search,
    )


def _stream_logs(
    *,
    args: Namespace,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> int:
    client = _client()
    logs_available = True
    poller = FollowLogPoller(
        start_ms=start_ms,
        end_ms=end_ms,
        dedupe_limit=_FOLLOW_LOG_DEDUPE_LIMIT,
        max_drain_polls=_FOLLOW_LOG_MAX_DRAIN_POLLS,
        max_retryable_errors=_FOLLOW_LOG_MAX_RETRYABLE_ERRORS,
    )
    status_retryable_error_count = 0

    while True:
        if not logs_available:
            time.sleep(_FOLLOW_LOG_POLL_INTERVAL_SECONDS)
            try:
                job_payload = client.cli_get_job(job_id=args.job_id)
            except MacrodataApiError as err:
                if not _is_retryable_api_error(err):
                    raise
                status_retryable_error_count += 1
                if status_retryable_error_count > _FOLLOW_LOG_MAX_RETRYABLE_ERRORS:
                    raise
                time.sleep(retry_delay(status_retryable_error_count))
                continue
            except MacrodataCredentialsError:
                raise
            status_retryable_error_count = 0
            logs_available = isinstance(job_payload.get("job"), dict) and bool(
                cast(dict[str, Any], job_payload["job"]).get("logsAvailable", True)
            )
            if _job_status(job_payload) in TERMINAL_JOB_STATUSES and not logs_available:
                return 0
            continue

        result = poller.poll(
            fetch_page=lambda page_start_ms, page_end_ms, page_cursor: _fetch_log_page(
                client=client,
                args=args,
                start_ms=page_start_ms,
                end_ms=page_end_ms,
                cursor=page_cursor,
                limit=limit,
            ),
            emit_entry=lambda entry: print(_format_log_entry(entry), flush=True),
            now_ms=_now_ms,
        )
        if result.action == "drain":
            time.sleep(_FOLLOW_LOG_DRAIN_POLL_DELAY_SECONDS)
            continue
        if result.action == "skip":
            _warn_follow_skip(
                start_ms=result.skipped_start_ms or poller.start_ms,
                end_ms=result.skipped_end_ms or poller.end_ms,
            )
            continue
        next_start_ms = result.next_window_start_ms or poller.start_ms
        try:
            job_payload = client.cli_get_job(job_id=args.job_id)
        except MacrodataApiError as err:
            if not _is_retryable_api_error(err):
                raise
            status_retryable_error_count += 1
            if status_retryable_error_count > _FOLLOW_LOG_MAX_RETRYABLE_ERRORS:
                raise
            time.sleep(retry_delay(status_retryable_error_count))
            continue
        except MacrodataCredentialsError:
            raise
        status_retryable_error_count = 0
        logs_available = isinstance(job_payload.get("job"), dict) and bool(
            cast(dict[str, Any], job_payload["job"]).get("logsAvailable", True)
        )
        if _job_status(job_payload) in TERMINAL_JOB_STATUSES:
            final_end_ms = max(_now_ms(), next_start_ms + 1)
            final_cursor: str | None = None
            final_retryable_error_count = 0
            seen_final_cursors: set[str] = set()
            while True:
                final_payload, final_retryable_error_count = call_with_retry(
                    lambda: _fetch_log_page(
                        client=client,
                        args=args,
                        start_ms=next_start_ms,
                        end_ms=final_end_ms,
                        cursor=final_cursor,
                        limit=limit,
                    ),
                    retryable_error_count=final_retryable_error_count,
                    max_retryable_errors=_FOLLOW_LOG_MAX_RETRYABLE_ERRORS,
                )
                final_retryable_error_count = 0
                entries = final_payload.get("entries")
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            print(_format_log_entry(entry), flush=True)
                next_final_cursor = _next_log_cursor(final_payload)
                if (
                    next_final_cursor is None
                    or next_final_cursor == final_cursor
                    or next_final_cursor in seen_final_cursors
                ):
                    break
                seen_final_cursors.add(next_final_cursor)
                final_cursor = next_final_cursor
            return 0
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
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return _render_logs(payload)
