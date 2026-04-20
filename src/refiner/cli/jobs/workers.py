from __future__ import annotations

from argparse import Namespace
from typing import Any

from refiner.cli.jobs.follow import format_ts as _format_ts
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import (
    _client,
    _dim_text,
    _print_next_command,
    _print_table,
    _run_job_command,
    _status_text,
    _timestamp_text,
)


def _stage_index_text(value: Any) -> str:
    stage_ref = _safe_text(value)
    if ":" not in stage_ref:
        return stage_ref
    return stage_ref.rsplit(":", 1)[-1]


def _workers_command_parts(args: Namespace) -> list[str]:
    parts = ["macrodata", "jobs", "workers", str(args.job_id)]
    if args.stage is not None:
        parts.extend(["--stage", str(args.stage)])
    if args.limit is not None:
        parts.extend(["--limit", str(args.limit)])
    if args.json:
        parts.append("--json")
    return parts


def _render_workers(payload: dict[str, Any], *, args: Namespace) -> int:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        print("No workers found.")
        return 0

    rows = [
        [
            _dim_text("ID"),
            _dim_text("Name"),
            _dim_text("Status"),
            _dim_text("Stage"),
            _dim_text("Shards Running"),
            _dim_text("Shards Done"),
            _dim_text("Started"),
            _dim_text("Ended"),
        ]
    ]
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                _safe_text(item.get("id")),
                _safe_text(item.get("name")),
                _status_text(item.get("status")),
                _dim_text(_stage_index_text(item.get("stageId"))),
                _safe_text(item.get("runningShardCount")),
                _safe_text(item.get("completedShardCount")),
                _timestamp_text(_format_ts(item.get("startedAt"))),
                _timestamp_text(_format_ts(item.get("endedAt"))),
            ]
        )
    _print_table(rows)
    page = payload.get("page")
    if isinstance(page, dict):
        _print_next_command(page.get("nextCursor"), _workers_command_parts(args))
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
        renderer=lambda payload: _render_workers(payload, args=args),
    )
