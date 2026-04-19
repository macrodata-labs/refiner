from __future__ import annotations

from collections.abc import Callable
import json
import sys
from typing import Any

from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError
from refiner.platform.client import MacrodataClient

_Payload = dict[str, Any]
_PayloadFetcher = Callable[[], _Payload]
_PayloadRenderer = Callable[[_Payload], int]


def _client() -> MacrodataClient:
    return MacrodataClient()


def _print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


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


def _handle_error(err: Exception) -> int:
    print(_safe_text(str(err)), file=sys.stderr)
    return 1


def _print_next_cursor(value: Any) -> None:
    if isinstance(value, str) and value:
        print(f"\nNext cursor: {_safe_text(value)}")


def _run_job_command(
    *,
    as_json: bool,
    fetch: _PayloadFetcher,
    renderer: _PayloadRenderer,
) -> int:
    try:
        payload = fetch()
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if as_json else renderer(payload)


def _executor_text(value: Any) -> str:
    if value == "cloud":
        return "cloud"
    if value == "local":
        return "local"
    return _safe_text(value)


def _progress_text(progress: Any) -> str:
    if not isinstance(progress, dict):
        return "-"
    done = progress.get("done")
    total = progress.get("total")
    if isinstance(done, int) and isinstance(total, int):
        return f"{done}/{total}"
    return "-"


def _started_by_text(item: dict[str, Any]) -> str:
    email = item.get("startedByEmail")
    username = item.get("startedByUsername")
    if isinstance(email, str) and email:
        if isinstance(username, str) and username:
            return _safe_text(f"{username} ({email})")
        return _safe_text(email)
    if isinstance(username, str) and username:
        return _safe_text(username)
    return "-"
