from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from refiner.platform.client import MacrodataApiError
from refiner.platform.client.api import sanitize_terminal_text

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "canceled"})


def safe_text(value: Any) -> str:
    if value is None:
        return "-"
    return sanitize_terminal_text(str(value))


def format_ts(value: Any) -> str:
    timestamp_ms = parse_epoch_ms(value)
    if timestamp_ms is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_epoch_ms(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    timestamp_ms = numeric * 1000 if numeric < 100_000_000_000 else numeric
    try:
        return int(timestamp_ms)
    except (TypeError, ValueError, OverflowError):
        return None


def job_status(payload: dict[str, Any]) -> str:
    job = payload.get("job")
    if not isinstance(job, dict):
        return ""
    return safe_text(job.get("status")).lower()


def log_entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    raw_message_hash = entry.get("messageHash")
    message_hash = (
        str(raw_message_hash)
        if isinstance(raw_message_hash, (str, int, float)) and str(raw_message_hash)
        else ""
    )
    return (
        safe_text(entry.get("ts")),
        safe_text(entry.get("workerId")),
        safe_text(entry.get("sourceType")),
        safe_text(entry.get("sourceName")),
        message_hash or safe_text(entry.get("line")),
    )


def remember_seen_key(
    *,
    key: tuple[str, str, str, str, str],
    seen_keys: set[tuple[str, str, str, str, str]],
    seen_order: deque[tuple[str, str, str, str, str]],
    limit: int,
) -> None:
    seen_keys.add(key)
    seen_order.append(key)
    while len(seen_order) > limit:
        seen_keys.discard(seen_order.popleft())


def is_retryable_api_error(err: Exception) -> bool:
    if isinstance(err, MacrodataApiError):
        return err.status in {0, 429} or err.status >= 500
    return False


def retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


def next_log_cursor(payload: dict[str, Any]) -> str | None:
    cursor = payload.get("nextCursor")
    return cursor if isinstance(cursor, str) and cursor else None


def epoch_ms_now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class FollowStatus:
    terminal_status: str | None = None
    logs_available: bool = True
    reset_window: bool = False


def follow_logs(
    *,
    start_ms: int,
    end_ms: int,
    poll_interval_seconds: float,
    status_interval_seconds: float,
    log_window_ms: int,
    max_drain_polls: int,
    drain_poll_delay_seconds: float,
    dedupe_limit: int,
    max_retryable_errors: int,
    fetch_logs: Callable[[int, int, str | None], dict[str, Any]],
    refresh_status: Callable[[], FollowStatus],
    emit_entry: Callable[[dict[str, Any]], bool],
    on_skip: Callable[[int, int], None],
    on_terminal: Callable[[str], None] | None = None,
    initial_status: FollowStatus | None = None,
    refresh_status_after_empty_page: bool = False,
    allow_status_during_drain: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    epoch_ms: Callable[[], int] = epoch_ms_now,
) -> int:
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    seen_order: deque[tuple[str, str, str, str, str]] = deque()
    current_start_ms = start_ms
    current_end_ms = end_ms
    current_cursor: str | None = None
    full_batch_polls = 0
    log_retryable_error_count = 0
    status_retryable_error_count = 0
    terminal_status = (
        initial_status.terminal_status if initial_status is not None else None
    )
    logs_available = (
        initial_status.logs_available if initial_status is not None else True
    )
    next_status_at = monotonic() + status_interval_seconds
    next_logs_at = 0.0

    def _apply_status(status: FollowStatus, now: float) -> None:
        nonlocal terminal_status, logs_available, current_cursor, full_batch_polls
        nonlocal current_start_ms, current_end_ms, next_logs_at, next_status_at
        terminal_status = status.terminal_status
        logs_available = status.logs_available
        if status.reset_window:
            current_cursor = None
            full_batch_polls = 0
            seen_keys.clear()
            seen_order.clear()
            reset_end_ms = epoch_ms()
            current_start_ms = max(0, reset_end_ms - log_window_ms)
            current_end_ms = max(current_start_ms + 1, reset_end_ms)
            next_logs_at = now
        next_status_at = now + status_interval_seconds

    while True:
        now = monotonic()
        if now > next_status_at:
            try:
                status = refresh_status()
            except Exception as err:
                if not is_retryable_api_error(err):
                    raise
                status_retryable_error_count += 1
                if status_retryable_error_count > max_retryable_errors:
                    raise
                next_status_at = now + retry_delay(status_retryable_error_count)
                continue
            status_retryable_error_count = 0
            _apply_status(status, now)

        if not logs_available and current_cursor is None:
            next_logs_at = now + poll_interval_seconds
        if (logs_available or current_cursor is not None) and now >= next_logs_at:
            try:
                payload = fetch_logs(current_start_ms, current_end_ms, current_cursor)
            except Exception as err:
                if not is_retryable_api_error(err):
                    raise
                log_retryable_error_count += 1
                if log_retryable_error_count > max_retryable_errors:
                    raise
                sleep(retry_delay(log_retryable_error_count))
                continue
            log_retryable_error_count = 0
            entries = payload.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    key = log_entry_key(entry)
                    if key in seen_keys:
                        continue
                    if not emit_entry(entry):
                        continue
                    remember_seen_key(
                        key=key,
                        seen_keys=seen_keys,
                        seen_order=seen_order,
                        limit=dedupe_limit,
                    )
            current_cursor = next_log_cursor(payload)
            if current_cursor is not None:
                full_batch_polls += 1
                if full_batch_polls < max_drain_polls:
                    if allow_status_during_drain:
                        next_logs_at = monotonic() + drain_poll_delay_seconds
                    else:
                        sleep(drain_poll_delay_seconds)
                        continue
                else:
                    skipped_end_ms = current_end_ms
                    oldest_entry = (
                        entries[0] if isinstance(entries, list) and entries else None
                    )
                    if isinstance(oldest_entry, dict):
                        parsed_oldest_ms = parse_epoch_ms(oldest_entry.get("ts"))
                        if parsed_oldest_ms is not None:
                            skipped_end_ms = parsed_oldest_ms
                    on_skip(current_start_ms, skipped_end_ms)
                    current_cursor = None
                    full_batch_polls = 0
                    current_start_ms = current_end_ms
                    current_end_ms = max(epoch_ms(), current_start_ms + 1)
                    next_logs_at = monotonic()
                    continue
            if current_cursor is not None:
                continue
            full_batch_polls = 0
            current_start_ms = current_end_ms
            current_end_ms = max(epoch_ms(), current_start_ms + 1)
            if refresh_status_after_empty_page and (
                not isinstance(entries, list) or not entries
            ):
                try:
                    status = refresh_status()
                except Exception as err:
                    if not is_retryable_api_error(err):
                        raise
                    status_retryable_error_count += 1
                    if status_retryable_error_count > max_retryable_errors:
                        raise
                    sleep(retry_delay(status_retryable_error_count))
                    continue
                status_retryable_error_count = 0
                _apply_status(status, now)
                if terminal_status is not None:
                    if on_terminal is not None:
                        on_terminal(terminal_status)
                    return 0
            next_logs_at = now + poll_interval_seconds

        if terminal_status is not None and current_cursor is None:
            if on_terminal is not None:
                on_terminal(terminal_status)
            return 0

        deadline = min(next_status_at, next_logs_at)
        sleep(max(0.05, deadline - monotonic()))
