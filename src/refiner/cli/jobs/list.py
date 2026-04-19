from __future__ import annotations

from argparse import Namespace
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _dim_text,
    _executor_text,
    _kind_text,
    _print_next_command,
    _print_table,
    _progress_text,
    _run_job_command,
    _started_by_text,
    _status_text,
    _timestamp_text,
    _value_text,
)


def _list_command_parts(args: Namespace) -> list[str]:
    parts = ["macrodata", "jobs", "list"]
    if args.status:
        parts.extend(["--status", str(args.status)])
    if args.kind:
        parts.extend(["--kind", str(args.kind)])
    if args.me:
        parts.append("--me")
    if args.limit is not None:
        parts.extend(["--limit", str(args.limit)])
    if args.json:
        parts.append("--json")
    return parts


def _render_list(payload: dict[str, Any], *, args: Namespace) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No jobs found.")
        return 0

    rows = [
        [
            _dim_text("ID"),
            _dim_text("Status"),
            _dim_text("Kind"),
            _dim_text("Name"),
            _dim_text("Stages"),
            _dim_text("Created"),
            _dim_text("Created By"),
        ]
    ]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _status_text(item.get("status")),
                _kind_text(_executor_text(item.get("executorKind"))),
                _value_text(item.get("name")),
                _progress_text(item.get("progress")),
                _timestamp_text(_format_ts(item.get("createdAt"))),
                _started_by_text(item),
            ]
        )
    _print_table(rows)
    _print_next_command(payload.get("nextCursor"), _list_command_parts(args))
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
        renderer=lambda payload: _render_list(payload, args=args),
    )
