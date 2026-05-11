from __future__ import annotations

import json
import re
import sys
from typing import Any

from refiner.cli.ui.terminal import stdout_is_interactive
from refiner.platform.client import MacrodataClient
from refiner.platform.client.api import sanitize_terminal_text

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_RESET = "\x1b[0m"
_DIM_COLOR = "\x1b[38;5;245m"


def create_client() -> MacrodataClient:
    return MacrodataClient()


def print_json(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def safe_text(value: Any) -> str:
    if value is None:
        return "-"
    return sanitize_terminal_text(str(value))


def dim_text(value: Any) -> str:
    text = safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_DIM_COLOR}{text}{_ANSI_RESET}"


def print_table(rows: list[list[str]]) -> None:
    if not rows:
        return
    column_count = len(rows[0])

    def visible_width(text: str) -> int:
        return len(_ANSI_RE.sub("", text))

    def pad_right(text: str, width: int) -> str:
        return text + (" " * max(0, width - visible_width(text)))

    widths = [
        max(visible_width(row[index]) if index < len(row) else 0 for row in rows)
        for index in range(column_count)
    ]
    for row_index, row in enumerate(rows):
        padded = "  ".join(
            pad_right(row[index] if index < len(row) else "", widths[index])
            for index in range(column_count)
        )
        print(padded.rstrip())
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def handle_error(err: Exception) -> int:
    print(safe_text(str(err)), file=sys.stderr)
    return 1
