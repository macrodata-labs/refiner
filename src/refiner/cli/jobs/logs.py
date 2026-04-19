from __future__ import annotations

from argparse import Namespace
from datetime import datetime as _real_datetime
from datetime import datetime, timezone
import json
from loguru import logger
import re
import sys
import time
from typing import Any, cast

from refiner.cli.jobs.follow import (
    ForwardLogPoller,
    TERMINAL_JOB_STATUSES,
    call_with_retry,
    emit_unique_entries,
    follow_skip_message,
    is_retryable_api_error as _is_retryable_api_error,
    job_status as _job_status,
    next_log_cursor as _next_log_cursor,
    parse_epoch_ms,
    retry_delay,
    safe_text as _safe_text,
)
from refiner.cli.jobs.common import _client, _handle_error
from refiner.cli.ui.terminal import stdout_is_interactive
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError

_WORKER_COLORS = (
    "\x1b[38;5;81m",
    "\x1b[38;5;75m",
    "\x1b[38;5;39m",
    "\x1b[38;5;44m",
    "\x1b[38;5;45m",
    "\x1b[38;5;51m",
    "\x1b[38;5;50m",
    "\x1b[38;5;49m",
    "\x1b[38;5;149m",
    "\x1b[38;5;114m",
    "\x1b[38;5;78m",
    "\x1b[38;5;84m",
    "\x1b[38;5;118m",
    "\x1b[38;5;154m",
    "\x1b[38;5;215m",
    "\x1b[38;5;208m",
    "\x1b[38;5;214m",
    "\x1b[38;5;179m",
    "\x1b[38;5;222m",
    "\x1b[38;5;141m",
    "\x1b[38;5;177m",
    "\x1b[38;5;183m",
    "\x1b[38;5;147m",
    "\x1b[38;5;140m",
    "\x1b[38;5;110m",
    "\x1b[38;5;117m",
    "\x1b[38;5;116m",
    "\x1b[38;5;109m",
    "\x1b[38;5;152m",
    "\x1b[38;5;221m",
    "\x1b[38;5;227m",
    "\x1b[38;5;186m",
    "\x1b[38;5;229m",
)
_TIMESTAMP_COLOR = "\x1b[38;5;255m"
_ANSI_RESET = "\x1b[0m"
_LOGURU_TAG_TO_ANSI = {
    "bold": "\x1b[1m",
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "BLACK": "\x1b[90m",
    "RED": "\x1b[91m",
    "GREEN": "\x1b[92m",
    "YELLOW": "\x1b[93m",
    "BLUE": "\x1b[94m",
    "MAGENTA": "\x1b[95m",
    "CYAN": "\x1b[96m",
    "WHITE": "\x1b[97m",
}
_LOGURU_LINE_RE = re.compile(
    r"^(?P<timestamp>[^|]+?) \| (?P<level>[A-Z]+)(?P<level_padding>\s*) \| (?P<rest>.*)$"
)
_LOGURU_TAG_RE = re.compile(r"<([^>]+)>")

_MAX_LOG_SEARCH_LIMIT = 100
_FOLLOW_LOG_POLL_INTERVAL_SECONDS = 1.0
_DEFAULT_LOG_PAGE_LIMIT = 100
_DEFAULT_FOLLOW_LOG_PAGE_LIMIT = 500
_FOLLOW_LOG_DEDUPE_LIMIT = 100_000
_FOLLOW_LOG_MAX_DRAIN_POLLS = 5
_FOLLOW_LOG_DRAIN_POLL_DELAY_SECONDS = 0.1
_FOLLOW_LOG_MAX_RETRYABLE_ERRORS = 5
_DISPLAY_WORKER_ID_HEAD_CHARS = 8
_DISPLAY_WORKER_ID_TAIL_CHARS = 6


def _now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def format_log_timestamp(value: Any) -> str:
    timestamp_ms = parse_epoch_ms(value)
    if timestamp_ms is None:
        return "-"
    try:
        dt = _real_datetime.fromtimestamp(timestamp_ms / 1000)
    except (OverflowError, OSError, ValueError):
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_log_level(severity: Any) -> str:
    raw = _safe_text(severity).strip().upper()
    if raw in {"WARN", "WARNING"}:
        return "WARNING"
    if raw in {"ERR", "ERROR"}:
        return "ERROR"
    if raw in {"DEBUG", "TRACE"}:
        return raw
    if raw in {"CRITICAL", "FATAL"}:
        return "CRITICAL"
    if raw:
        return raw
    return "INFO"


def format_log_source(
    *,
    source_type: Any,
    source_name: Any,
    default: str = "",
) -> str:
    normalized_type = _safe_text(source_type)
    normalized_name = _safe_text(source_name)
    if normalized_type == "worker" and normalized_name in {"-", "worker"}:
        return ""
    if (
        normalized_type != "-"
        and normalized_name != "-"
        and normalized_type != normalized_name
    ):
        return f"{normalized_type}:{normalized_name}"
    if normalized_name != "-":
        return normalized_name
    if normalized_type != "-":
        return normalized_type
    return default


def format_log_worker_label(
    *,
    worker_id: Any,
    source_name: Any = None,
    default: str = "cloud",
) -> str:
    normalized_worker_id = _safe_text(worker_id).strip()
    if normalized_worker_id and normalized_worker_id != "-":
        if len(normalized_worker_id) <= (
            _DISPLAY_WORKER_ID_HEAD_CHARS + _DISPLAY_WORKER_ID_TAIL_CHARS + 1
        ):
            return normalized_worker_id
        return (
            f"{normalized_worker_id[:_DISPLAY_WORKER_ID_HEAD_CHARS]}..."
            f"{normalized_worker_id[-_DISPLAY_WORKER_ID_TAIL_CHARS:]}"
        )
    normalized_source_name = _safe_text(source_name).strip()
    if normalized_source_name and normalized_source_name != "-":
        return normalized_source_name
    return default


def _loguru_markup_to_ansi(markup: str) -> str:
    ansi_parts: list[str] = []
    for tag in _LOGURU_TAG_RE.findall(markup):
        ansi = _LOGURU_TAG_TO_ANSI.get(tag)
        if ansi is not None:
            ansi_parts.append(ansi)
    return "".join(ansi_parts)


def format_rendered_log_line(*, worker_id: str, line: str, interactive: bool) -> str:
    prefix = f"worker={worker_id}"
    if not interactive:
        return f"{prefix} {line}"
    worker_color = _WORKER_COLORS[sum(worker_id.encode("utf-8")) % len(_WORKER_COLORS)]
    match = _LOGURU_LINE_RE.match(line)
    if match is None:
        return f"{worker_color}{prefix} {line}{_ANSI_RESET}"
    level = match.group("level")
    return (
        f"{worker_color}{prefix}{_ANSI_RESET} "
        f"{_TIMESTAMP_COLOR}{match.group('timestamp').strip()}{_ANSI_RESET} | "
        f"{_loguru_markup_to_ansi(logger.level(level).color) or worker_color}"
        f"{level}{match.group('level_padding')}{_ANSI_RESET} | "
        f"{worker_color}{match.group('rest')}{_ANSI_RESET}"
    )


def _format_log_entry(entry: dict[str, Any]) -> str:
    level = format_log_level(entry.get("severity"))
    source = format_log_source(
        source_type=entry.get("sourceType"),
        source_name=entry.get("sourceName"),
    )
    source_text = f"{source} - " if source else ""
    line = f"{format_log_timestamp(entry.get('ts'))} | {level.ljust(8)} | {source_text}{_safe_text(entry.get('line'))}"
    return format_rendered_log_line(
        worker_id=format_log_worker_label(
            worker_id=entry.get("workerId"),
            source_name=entry.get("sourceName"),
            default="worker",
        ),
        line=line,
        interactive=stdout_is_interactive(),
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
        print("\nMore logs are available after these results.")
    return 0


def _fetch_log_page(
    *,
    client: Any,
    args: Namespace,
    start_ms: int | None,
    end_ms: int | None,
    cursor: str | None,
    limit: int,
    anchor: str,
) -> dict[str, Any]:
    return client.cli_get_job_logs(
        job_id=args.job_id,
        start_ms=start_ms,
        end_ms=end_ms,
        anchor=anchor,
        cursor=cursor,
        limit=limit,
        stage_index=args.stage,
        worker_id=args.worker,
        source_type=args.source_type,
        source_name=args.source_name,
        severity=args.severity,
        search=args.search,
    )


def _terminal_entry_sort_key(entry: dict[str, Any]) -> tuple[int, str, str, str, str]:
    return (
        parse_epoch_ms(entry.get("ts")) or 0,
        _safe_text(entry.get("workerId")),
        _safe_text(entry.get("sourceType")),
        _safe_text(entry.get("sourceName")),
        _safe_text(entry.get("messageHash")) or _safe_text(entry.get("line")),
    )


def _bootstrap_latest_tail(
    *,
    client: Any,
    args: Namespace,
    limit: int,
    emit_entry: Any,
    seen_keys: set[tuple[str, str, str, str, str]],
    seen_order: Any,
    dedupe_limit: int,
) -> int:
    payload = _fetch_log_page(
        client=client,
        args=args,
        start_ms=None,
        end_ms=None,
        cursor=None,
        limit=limit,
        anchor="latest",
    )
    _, newest_emitted_ms = emit_unique_entries(
        entries=payload.get("entries"),
        emit_entry=emit_entry,
        seen_keys=seen_keys,
        seen_order=seen_order,
        limit=dedupe_limit,
    )
    return newest_emitted_ms or _now_ms()


def _stream_logs(
    *,
    args: Namespace,
    limit: int,
) -> int:
    client = _client()
    logs_available = True
    poller = ForwardLogPoller(
        start_ms=_now_ms(),
        dedupe_limit=_FOLLOW_LOG_DEDUPE_LIMIT,
        max_drain_polls=_FOLLOW_LOG_MAX_DRAIN_POLLS,
        max_retryable_errors=_FOLLOW_LOG_MAX_RETRYABLE_ERRORS,
    )
    status_retryable_error_count = 0
    bootstrap_done = False

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

        if not bootstrap_done:
            bootstrap_start_ms, poller.retryable_error_count = call_with_retry(
                lambda: _bootstrap_latest_tail(
                    client=client,
                    args=args,
                    limit=limit,
                    emit_entry=lambda entry: print(
                        _format_log_entry(entry), flush=True
                    ),
                    seen_keys=poller.seen_keys,
                    seen_order=poller.seen_order,
                    dedupe_limit=poller.dedupe_limit,
                ),
                retryable_error_count=poller.retryable_error_count,
                max_retryable_errors=poller.max_retryable_errors,
            )
            poller.retryable_error_count = 0
            poller.reset_window(start_ms=bootstrap_start_ms)
            bootstrap_done = True

        result = poller.poll(
            fetch_page=lambda page_start_ms, page_end_ms, page_cursor: _fetch_log_page(
                client=client,
                args=args,
                start_ms=page_start_ms,
                end_ms=page_end_ms,
                cursor=page_cursor,
                limit=limit,
                anchor="earliest",
            ),
            emit_entry=lambda entry: print(_format_log_entry(entry), flush=True),
            now_ms=_now_ms,
        )
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
            final_entries: list[dict[str, Any]] = []
            while True:
                final_payload, final_retryable_error_count = call_with_retry(
                    lambda: _fetch_log_page(
                        client=client,
                        args=args,
                        start_ms=next_start_ms,
                        end_ms=final_end_ms,
                        cursor=final_cursor,
                        limit=limit,
                        anchor="earliest",
                    ),
                    retryable_error_count=final_retryable_error_count,
                    max_retryable_errors=_FOLLOW_LOG_MAX_RETRYABLE_ERRORS,
                )
                final_retryable_error_count = 0
                emit_unique_entries(
                    entries=final_payload.get("entries"),
                    emit_entry=final_entries.append,
                    seen_keys=poller.seen_keys,
                    seen_order=poller.seen_order,
                    limit=poller.dedupe_limit,
                )
                next_final_cursor = _next_log_cursor(final_payload)
                if (
                    next_final_cursor is None
                    or next_final_cursor == final_cursor
                    or next_final_cursor in seen_final_cursors
                ):
                    break
                seen_final_cursors.add(next_final_cursor)
                final_cursor = next_final_cursor
            for entry in sorted(final_entries, key=_terminal_entry_sort_key):
                print(_format_log_entry(entry), flush=True)
            return 0
        if result.action == "drain":
            time.sleep(_FOLLOW_LOG_DRAIN_POLL_DELAY_SECONDS)
            continue
        if result.action == "skip":
            _warn_follow_skip(
                start_ms=result.skipped_start_ms or poller.start_ms,
                end_ms=result.skipped_end_ms or next_start_ms,
            )
            continue
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
        anchor = "earliest"
    else:
        limit = _effective_log_limit(args)
        if args.start_ms is None and args.end_ms is None and args.cursor is None:
            start_ms = 0
            end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            anchor = "latest"
        else:
            end_ms = (
                args.end_ms
                if args.end_ms is not None
                else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            )
            start_ms = (
                args.start_ms if args.start_ms is not None else end_ms - 60 * 60 * 1000
            )
            anchor = "earliest"

    if args.follow:
        try:
            startup_job_payload = _client().cli_get_job(job_id=args.job_id)
        except KeyboardInterrupt:
            return 130
        except (MacrodataApiError, MacrodataCredentialsError) as err:
            if isinstance(err, MacrodataApiError) and _is_retryable_api_error(err):
                startup_job_payload = None
            else:
                return _handle_error(err)
        if (
            isinstance(startup_job_payload, dict)
            and _job_status(startup_job_payload) in TERMINAL_JOB_STATUSES
        ):
            print(
                "job is already terminal; showing one-shot logs instead of follow mode.",
                file=sys.stderr,
            )
            args = Namespace(**{**vars(args), "follow": False})
        else:
            try:
                return _stream_logs(
                    args=args,
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
            anchor=anchor,
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
