from __future__ import annotations

import json
import sys
from typing import Any

from refiner.cli.job_utils import safe_text as _safe_text
from refiner.platform.client import MacrodataClient


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


def _step_summary_text(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return "-"
    parts: list[str] = []
    for key in sorted(args.keys())[:3]:
        value = args.get(key)
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
        elif isinstance(value, list):
            parts.append(f"{key}=[{len(value)}]")
        elif isinstance(value, dict):
            parts.append(f"{key}={{...}}")
    return _safe_text(", ".join(parts) if parts else "{...}")
