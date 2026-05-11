from __future__ import annotations

from collections.abc import Callable
import shlex
from typing import Any

from refiner.cli.common import handle_error
from refiner.cli.common import print_json
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.ui.terminal import stdout_is_interactive
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError

_Payload = dict[str, Any]
_PayloadFetcher = Callable[[], _Payload]
_PayloadRenderer = Callable[[_Payload], int]
_ANSI_RESET = "\x1b[0m"
_STATUS_COLORS = {
    "pending": "\x1b[1;38;5;110m",
    "running": "\x1b[1;38;5;220m",
    "completed": "\x1b[1;38;5;77m",
    "failed": "\x1b[1;38;5;203m",
    "canceled": "\x1b[1;38;5;245m",
}
_STATUS_DOT = "●"
_DIM_COLOR = "\x1b[38;5;245m"
_TIMESTAMP_COLOR = "\x1b[38;5;255m"
_VALUE_COLOR = "\x1b[1;38;5;255m"
_SECTION_COLOR = "\x1b[1;38;5;117m"
_ERROR_COLOR = "\x1b[1;38;5;203m"
_KIND_COLORS = {
    "cloud": "\x1b[1;38;5;117m",
    "local": "\x1b[1;38;5;214m",
}
_LEVEL_COLORS = {
    "DEBUG": "\x1b[38;5;245m",
    "INFO": "\x1b[1;38;5;77m",
    "WARNING": "\x1b[1;38;5;220m",
    "ERROR": "\x1b[1;38;5;203m",
    "CRITICAL": "\x1b[1;38;5;203m",
}


def _status_text(value: Any) -> str:
    status = _safe_text(value)
    if not stdout_is_interactive():
        return status
    color = _STATUS_COLORS.get(status.lower())
    if color is None:
        return status
    return f"{color}{_STATUS_DOT} {status}{_ANSI_RESET}"


def _dim_text(value: Any) -> str:
    text = _safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_DIM_COLOR}{text}{_ANSI_RESET}"


def _level_text(value: Any) -> str:
    level = _safe_text(value).strip().upper() or "INFO"
    if not stdout_is_interactive():
        return level
    color = _LEVEL_COLORS.get(level, _DIM_COLOR)
    return f"{color}{level}{_ANSI_RESET}"


def _timestamp_text(value: Any) -> str:
    text = _safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_TIMESTAMP_COLOR}{text}{_ANSI_RESET}"


def _kind_text(value: Any) -> str:
    kind = _safe_text(value)
    if not stdout_is_interactive():
        return kind
    color = _KIND_COLORS.get(kind.lower())
    if color is None:
        return kind
    return f"{color}{kind}{_ANSI_RESET}"


def _value_text(value: Any) -> str:
    text = _safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_VALUE_COLOR}{text}{_ANSI_RESET}"


def _section_text(value: Any) -> str:
    text = _safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_SECTION_COLOR}{text}{_ANSI_RESET}"


def _label_text(value: Any) -> str:
    return _dim_text(value)


def _error_text(value: Any) -> str:
    text = _safe_text(value)
    if not stdout_is_interactive():
        return text
    return f"{_ERROR_COLOR}{text}{_ANSI_RESET}"


def _shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _print_next_command(cursor: Any, parts: list[str]) -> None:
    if isinstance(cursor, str) and cursor:
        command = _shell_command([*parts, "--cursor", _safe_text(cursor)])
        print(f"\nNext cursor: {command}")


def _run_job_command(
    *,
    as_json: bool,
    fetch: _PayloadFetcher,
    renderer: _PayloadRenderer,
) -> int:
    try:
        payload = fetch()
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return handle_error(err)
    return print_json(payload) if as_json else renderer(payload)


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
