from __future__ import annotations

from collections import deque
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


def retry_delay(error_count: int) -> float:
    return min(float(2 ** max(0, error_count - 1)), 5.0)


def is_retryable_api_error(err: Exception) -> bool:
    if isinstance(err, MacrodataApiError):
        return err.status in {0, 429} or err.status >= 500
    return False


def next_log_cursor(payload: dict[str, Any]) -> str | None:
    cursor = payload.get("nextCursor")
    return cursor if isinstance(cursor, str) and cursor else None
