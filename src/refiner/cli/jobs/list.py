from __future__ import annotations

from argparse import Namespace
from typing import Any

from refiner.cli.job_utils import format_ts as _format_ts
from refiner.cli.job_utils import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _executor_text,
    _print_next_cursor,
    _print_table,
    _progress_text,
    _run_job_command,
)


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
    _print_next_cursor(payload.get("nextCursor"))
    return 0


def cmd_jobs_list(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_list_jobs(
            status=args.status,
            executor_kind=args.kind,
            me=args.me,
            limit=args.limit,
            cursor=args.cursor,
        ),
        renderer=_render_list,
    )
