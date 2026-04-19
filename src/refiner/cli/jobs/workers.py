from __future__ import annotations

from argparse import Namespace
from typing import Any

from refiner.cli.job_utils import format_ts as _format_ts
from refiner.cli.job_utils import safe_text as _safe_text
from refiner.cli.jobs.common import _client, _handle_error, _print_json, _print_table
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError


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
        next_cursor = page.get("nextCursor")
        if isinstance(next_cursor, str) and next_cursor:
            print(f"\nNext cursor: {_safe_text(next_cursor)}")
    return 0


def cmd_jobs_workers(args: Namespace) -> int:
    try:
        payload = _client().cli_get_job_workers(
            job_id=args.job_id,
            stage_index=args.stage,
            limit=args.limit,
            cursor=args.cursor,
        )
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_workers(payload)
