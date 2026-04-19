from __future__ import annotations

from argparse import Namespace
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _print_next_cursor,
    _print_table,
    _run_job_command,
)


def _render_workers(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No workers found.")
        return 0

    rows = [["UUID", "Name", "Status", "Stage", "Running", "Done", "Started", "Host"]]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("name")),
                _safe_text(item.get("status")),
                _safe_text(item.get("stageId")),
                _safe_text(item.get("runningShardCount")),
                _safe_text(item.get("completedShardCount")),
                _format_ts(item.get("startedAt")),
                _safe_text(item.get("host")),
            ]
        )
    _print_table(rows)
    page = payload.get("page")
    if isinstance(page, dict):
        _print_next_cursor(page.get("nextCursor"))
    return 0


def cmd_jobs_workers(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_get_job_workers(
            job_id=args.job_id,
            stage_index=args.stage,
            limit=args.limit,
            cursor=args.cursor,
        ),
        renderer=_render_workers,
    )
