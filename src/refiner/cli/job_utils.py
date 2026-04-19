from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Callable, TypeVar

from refiner.platform.client import MacrodataApiError
from refiner.platform.client.api import sanitize_terminal_text

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "canceled"})
T = TypeVar("T")
_LogKey = tuple[str, str, str, str, str]


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


def retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


def is_retryable_api_error(err: Exception) -> bool:
    if isinstance(err, MacrodataApiError):
        return err.status in {0, 429} or err.status >= 500
    return False


def next_log_cursor(payload: dict[str, Any]) -> str | None:
    cursor = payload.get("nextCursor")
    return cursor if isinstance(cursor, str) and cursor else None


def follow_skip_message(*, start_ms: int, end_ms: int) -> str:
    return (
        "log volume is high; skipped older backlog from "
        f"{format_ts(start_ms)} to {format_ts(end_ms)} to stay live"
    )


def call_with_retry(
    operation: Callable[[], T],
    *,
    retryable_error_count: int,
    max_retryable_errors: int,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    while True:
        try:
            return operation(), retryable_error_count
        except Exception as err:
            if not is_retryable_api_error(err):
                raise
            retryable_error_count += 1
            if retryable_error_count > max_retryable_errors:
                raise
            sleep_fn(retry_delay(retryable_error_count))


@dataclass
class FollowPollResult:
    action: str
    next_window_start_ms: int | None = None
    skipped_start_ms: int | None = None
    skipped_end_ms: int | None = None


@dataclass
class FollowLogPoller:
    start_ms: int
    end_ms: int
    dedupe_limit: int
    max_drain_polls: int
    max_retryable_errors: int
    cursor: str | None = None
    full_batch_polls: int = 0
    retryable_error_count: int = 0
    seen_keys: set[_LogKey] = field(default_factory=set)
    seen_order: deque[_LogKey] = field(default_factory=deque)

    def reset_window(self, *, now_ms: int, window_ms: int) -> None:
        self.start_ms = max(0, now_ms - window_ms)
        self.end_ms = max(self.start_ms + 1, now_ms)
        self.cursor = None
        self.full_batch_polls = 0
        self.retryable_error_count = 0
        self.seen_keys.clear()
        self.seen_order.clear()

    def poll(
        self,
        *,
        fetch_page: Callable[[int, int, str | None], dict[str, Any]],
        emit_entry: Callable[[dict[str, Any]], None],
        now_ms: Callable[[], int],
    ) -> FollowPollResult:
        payload, self.retryable_error_count = call_with_retry(
            lambda: fetch_page(self.start_ms, self.end_ms, self.cursor),
            retryable_error_count=self.retryable_error_count,
            max_retryable_errors=self.max_retryable_errors,
        )
        self.retryable_error_count = 0
        oldest_emitted_ms: int | None = None
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = log_entry_key(entry)
                if key in self.seen_keys:
                    continue
                emit_entry(entry)
                remember_seen_key(
                    key=key,
                    seen_keys=self.seen_keys,
                    seen_order=self.seen_order,
                    limit=self.dedupe_limit,
                )
                parsed_entry_ms = parse_epoch_ms(entry.get("ts"))
                if parsed_entry_ms is not None and oldest_emitted_ms is None:
                    oldest_emitted_ms = parsed_entry_ms

        self.cursor = next_log_cursor(payload)
        if self.cursor is not None:
            self.full_batch_polls += 1
            if self.full_batch_polls < self.max_drain_polls:
                return FollowPollResult(action="drain")
            skipped_start_ms = self.start_ms
            skipped_end_ms = oldest_emitted_ms or self.end_ms
            self.cursor = None
            self.full_batch_polls = 0
            self.start_ms = self.end_ms
            self.end_ms = max(now_ms(), self.start_ms + 1)
            return FollowPollResult(
                action="skip",
                skipped_start_ms=skipped_start_ms,
                skipped_end_ms=skipped_end_ms,
            )

        self.full_batch_polls = 0
        next_window_start_ms = self.end_ms
        next_window_end_ms = max(now_ms(), next_window_start_ms + 1)
        self.start_ms = next_window_start_ms
        self.end_ms = next_window_end_ms
        return FollowPollResult(
            action="advance",
            next_window_start_ms=next_window_start_ms,
        )
