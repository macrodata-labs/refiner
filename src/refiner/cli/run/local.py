from __future__ import annotations

import io
import json
import os
import re
import shutil
import select
import subprocess
import sys
import time
import ctypes
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from loguru import logger
from refiner.cli.ui import stdout_is_interactive

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_WORKER_COLORS = (
    "\x1b[38;5;81m",
    "\x1b[38;5;75m",
    "\x1b[38;5;39m",
    "\x1b[38;5;44m",
    "\x1b[38;5;45m",
    "\x1b[38;5;51m",
    "\x1b[38;5;50m",
    "\x1b[38;5;49m",
    "\x1b[38;5;149m",
    "\x1b[38;5;114m",
    "\x1b[38;5;78m",
    "\x1b[38;5;84m",
    "\x1b[38;5;118m",
    "\x1b[38;5;154m",
    "\x1b[38;5;215m",
    "\x1b[38;5;208m",
    "\x1b[38;5;214m",
    "\x1b[38;5;179m",
    "\x1b[38;5;222m",
    "\x1b[38;5;141m",
    "\x1b[38;5;177m",
    "\x1b[38;5;183m",
    "\x1b[38;5;147m",
    "\x1b[38;5;140m",
    "\x1b[38;5;110m",
    "\x1b[38;5;117m",
    "\x1b[38;5;116m",
    "\x1b[38;5;109m",
    "\x1b[38;5;152m",
    "\x1b[38;5;221m",
    "\x1b[38;5;227m",
    "\x1b[38;5;186m",
    "\x1b[38;5;229m",
)
_TIMESTAMP_COLOR = "\x1b[38;5;255m"
_TITLE_COLOR = "\x1b[1;38;5;45m"
_LABEL_COLOR = "\x1b[38;5;110m"
_VALUE_COLOR = "\x1b[1;38;5;255m"
_URL_COLOR = "\x1b[4;38;5;117m"
_SEPARATOR_COLOR = "\x1b[38;5;239m"
_STATUS_COLORS = {
    "running": "\x1b[1;38;5;220m",
    "completed": "\x1b[1;38;5;77m",
    "failed": "\x1b[1;38;5;203m",
}
_LOGURU_TAG_TO_ANSI = {
    "bold": "\x1b[1m",
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "BLACK": "\x1b[90m",
    "RED": "\x1b[91m",
    "GREEN": "\x1b[92m",
    "YELLOW": "\x1b[93m",
    "BLUE": "\x1b[94m",
    "MAGENTA": "\x1b[95m",
    "CYAN": "\x1b[96m",
    "WHITE": "\x1b[97m",
}
_ANSI_RESET = "\x1b[0m"
_LOGURU_LINE_RE = re.compile(
    r"^(?P<timestamp>[^|]+?) \| (?P<level>[A-Z]+)(?P<level_padding>\s*) \| (?P<rest>.*)$"
)
_LOGURU_TAG_RE = re.compile(r"<([^>]+)>")
_VALID_LOG_MODES = {"all", "none", "one", "errors"}
_LOCAL_LOG_MODE_ENV_VAR = "REFINER_LOCAL_LOGS"


class LocalLaunchResumeError(RuntimeError):
    pass


class LocalLaunchInterrupted(KeyboardInterrupt):
    pass


@dataclass(frozen=True, slots=True)
class LaunchStats:
    job_id: str
    workers: int
    claimed: int
    completed: int
    failed: int
    output_rows: int


@dataclass(frozen=True, slots=True)
class LocalStageSnapshot:
    job_id: str
    job_name: str
    rundir: str | None
    stage_index: int
    total_stages: int
    stage_workers: int
    tracking_url: str | None
    status: str
    worker_total: int
    worker_running: int
    worker_completed: int
    worker_failed: int
    elapsed_seconds: float


@dataclass(slots=True)
class WorkerProcessMonitor:
    worker_id: str
    process: subprocess.Popen[str]
    stdout_buffer: list[str]
    stderr_buffer: list[str]
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread


def _loguru_markup_to_ansi(markup: str) -> str:
    ansi_parts: list[str] = []
    for tag in _LOGURU_TAG_RE.findall(markup):
        ansi = _LOGURU_TAG_TO_ANSI.get(tag)
        if ansi is not None:
            ansi_parts.append(ansi)
    return "".join(ansi_parts)


def _visible_width(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _pad_right(text: str, width: int) -> str:
    padding = max(0, width - _visible_width(text))
    return text + (" " * padding)


def _truncate_plain(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def normalize_log_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _VALID_LOG_MODES:
        allowed = ", ".join(sorted(_VALID_LOG_MODES))
        raise ValueError(
            f"unsupported local log mode {mode!r}; expected one of: {allowed}"
        )
    return normalized


def resolve_log_mode(mode: str | None) -> str:
    if mode is not None:
        return normalize_log_mode(mode)
    env_mode = os.environ.get(_LOCAL_LOG_MODE_ENV_VAR)
    if env_mode:
        return normalize_log_mode(env_mode)
    return "all"


def should_emit_worker_line(
    *,
    log_mode: str,
    worker_id: str,
    selected_worker_id: str | None,
    line: str,
    severity: Any = None,
) -> bool:
    if log_mode == "all":
        return True
    if log_mode == "none":
        return False
    if log_mode == "one":
        return selected_worker_id is None or worker_id == selected_worker_id
    if log_mode == "errors":
        if isinstance(severity, str):
            return severity.strip().lower() == "error"
        match = _LOGURU_LINE_RE.match(line)
        return match is not None and match.group("level").upper() in {
            "ERROR",
            "CRITICAL",
        }
    return True


def format_resume_message(message: str, *, rundir: str | None) -> str:
    suffix = (
        f" To resume completed shards, rerun with rundir={rundir!r}."
        if rundir is not None
        else ""
    )
    return message.rstrip(".") + "." + suffix


def stop_worker_monitors(
    monitors: list[WorkerProcessMonitor],
    *,
    terminate_timeout_seconds: float,
) -> None:
    running = [monitor for monitor in monitors if monitor.process.poll() is None]
    for monitor in running:
        try:
            monitor.process.terminate()
        except Exception:
            continue
    deadline = time.monotonic() + terminate_timeout_seconds
    for monitor in running:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            monitor.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                monitor.process.kill()
            except Exception:
                pass
    for monitor in monitors:
        for thread in (monitor.stdout_thread, monitor.stderr_thread):
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        try:
            monitor.process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass


def collect_local_stage_results(
    *,
    job_id: str,
    job_name: str,
    rundir: str,
    stage_index: int,
    total_stages: int,
    stage_workers: int,
    tracking_url: str | None,
    processes: list[tuple[str, subprocess.Popen[str]]],
    log_mode: str | None,
    interrupt_message: str,
    terminate_timeout_seconds: float,
) -> LaunchStats:
    monitors: list[WorkerProcessMonitor] = []
    for worker_id, process in processes:
        stdout_buffer: list[str] = []
        stderr_buffer: list[str] = []
        stdout_thread = threading.Thread(
            target=_drain_stream,
            kwargs={"stream": process.stdout, "target": stdout_buffer},
            name=f"refiner-worker-stdout-{worker_id}",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_stream,
            kwargs={"stream": process.stderr, "target": stderr_buffer},
            name=f"refiner-worker-stderr-{worker_id}",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        monitors.append(
            WorkerProcessMonitor(
                worker_id=worker_id,
                process=process,
                stdout_buffer=stdout_buffer,
                stderr_buffer=stderr_buffer,
                stdout_thread=stdout_thread,
                stderr_thread=stderr_thread,
            )
        )
    worker_log_paths = {
        worker_id: Path(rundir)
        / f"stage-{stage_index}"
        / "logs"
        / f"worker-{worker_id}.log"
        for worker_id, _ in processes
    }
    stage_started_at = time.monotonic()

    def snapshot() -> LocalStageSnapshot:
        completed = 0
        failed = 0
        for monitor in monitors:
            returncode = monitor.process.poll()
            if returncode is None:
                continue
            if returncode == 0:
                completed += 1
            else:
                failed += 1
        running = max(0, len(monitors) - completed - failed)
        return LocalStageSnapshot(
            job_id=job_id,
            job_name=job_name,
            rundir=rundir,
            stage_index=stage_index,
            total_stages=total_stages,
            stage_workers=stage_workers,
            tracking_url=tracking_url,
            status="running" if running else ("failed" if failed > 0 else "completed"),
            worker_total=len(monitors),
            worker_running=running,
            worker_completed=completed,
            worker_failed=failed,
            elapsed_seconds=time.monotonic() - stage_started_at,
        )

    try:
        try:
            resolved_log_mode = resolve_log_mode(log_mode)
        except ValueError as err:
            raise SystemExit(str(err)) from err
        run_local_stage_ui(
            worker_log_paths=worker_log_paths,
            snapshot_getter=snapshot,
            log_mode=resolved_log_mode,
            interrupt_message=interrupt_message,
        )
        errors: list[str] = []
        claimed = 0
        completed = 0
        failed = 0
        output_rows = 0
        for monitor in monitors:
            monitor.stdout_thread.join()
            monitor.stderr_thread.join()
            monitor.process.wait()
            stdout = "".join(monitor.stdout_buffer)
            stderr = "".join(monitor.stderr_buffer)
            final_stdout_line = next(
                (line for line in reversed(stdout.splitlines()) if line.strip()),
                "",
            )
            try:
                decoded = json.loads(final_stdout_line or "{}")
            except json.JSONDecodeError:
                decoded = None
            raw = (
                decoded
                if isinstance(decoded, dict)
                else {
                    "worker_id": monitor.worker_id,
                    "claimed": 0,
                    "completed": 0,
                    "failed": 1,
                    "output_rows": 0,
                    "error": (
                        stderr.strip()
                        or stdout.strip()
                        or f"worker process exited with code {monitor.process.returncode}"
                    ),
                }
            )
            if raw.get("error") is not None:
                errors.append(f"worker {raw.get('worker_id', '')}: {raw['error']}")
            if monitor.process.returncode not in (0, None):
                errors.append(
                    f"worker process exited with code {monitor.process.returncode}"
                )
            parsed_claimed = raw.get("claimed", 0)
            parsed_completed = raw.get("completed", 0)
            parsed_failed = raw.get("failed", 0)
            parsed_output_rows = raw.get("output_rows", 0)
            claimed += (
                int(parsed_claimed)
                if isinstance(parsed_claimed, int | float | str)
                else 0
            )
            completed += (
                int(parsed_completed)
                if isinstance(parsed_completed, int | float | str)
                else 0
            )
            failed += (
                int(parsed_failed)
                if isinstance(parsed_failed, int | float | str)
                else 0
            )
            output_rows += (
                int(parsed_output_rows)
                if isinstance(parsed_output_rows, int | float | str)
                else 0
            )
        if errors:
            raise RuntimeError("; ".join(sorted(set(errors))))
        return LaunchStats(
            job_id=job_id,
            workers=stage_workers,
            claimed=claimed,
            completed=completed,
            failed=failed,
            output_rows=output_rows,
        )
    except BaseException:
        stop_worker_monitors(
            monitors,
            terminate_timeout_seconds=terminate_timeout_seconds,
        )
        raise


class LocalStageLogTail:
    _READ_CHUNK_BYTES = 1024 * 1024

    def __init__(self, *, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._partial = ""
        self._handle: io.TextIOWrapper | None = None

    def poll(self) -> list[str]:
        if self._handle is None:
            if not self.path.exists():
                return []
            self._handle = self.path.open(encoding="utf-8")
            self._handle.seek(self._offset)
        chunk = self._handle.read(self._READ_CHUNK_BYTES)
        self._offset = self._handle.tell()
        if not chunk:
            return []
        data = self._partial + chunk
        lines = data.splitlines(keepends=True)
        emitted: list[str] = []
        self._partial = ""
        for line in lines:
            if line.endswith("\n"):
                emitted.append(line.rstrip("\n"))
            else:
                self._partial = line
        return emitted

    def flush(self) -> list[str]:
        lines = self.poll()
        if self._partial:
            lines.append(self._partial)
            self._partial = ""
        return lines

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class LocalStageLogWatcher:
    _INOTIFY_EVENT_BUFFER_BYTES = 4096
    _IN_MODIFY = 0x00000002
    _IN_CREATE = 0x00000100
    _IN_MOVED_TO = 0x00000080
    _IN_CLOSE_WRITE = 0x00000008
    _IN_ATTRIB = 0x00000004

    def __init__(self, *, paths: dict[str, Path]) -> None:
        self._directory = next(iter(paths.values())).parent if paths else None
        self._inotify_fd: int | None = None
        self._watch_descriptor: int | None = None
        self._libc: ctypes.CDLL | None = None
        self._init_inotify()

    def wait(self, timeout_seconds: float) -> None:
        if self._inotify_fd is None:
            self._init_inotify()
        if self._inotify_fd is None:
            time.sleep(timeout_seconds)
            return
        ready, _, _ = select.select([self._inotify_fd], [], [], timeout_seconds)
        if not ready:
            return
        try:
            os.read(self._inotify_fd, self._INOTIFY_EVENT_BUFFER_BYTES)
        except BlockingIOError:
            return

    def close(self) -> None:
        if self._watch_descriptor is not None and self._inotify_fd is not None:
            try:
                assert self._libc is not None
                self._libc.inotify_rm_watch(self._inotify_fd, self._watch_descriptor)
            except Exception:
                pass
        if self._inotify_fd is not None:
            try:
                os.close(self._inotify_fd)
            except OSError:
                pass
        self._watch_descriptor = None
        self._inotify_fd = None
        self._libc = None

    def _init_inotify(self) -> None:
        if (
            self._directory is None
            or self._inotify_fd is not None
            or os.name != "posix"
            or not self._directory.exists()
        ):
            return
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if fd < 0:
                return
            mask = (
                self._IN_MODIFY
                | self._IN_CREATE
                | self._IN_MOVED_TO
                | self._IN_CLOSE_WRITE
                | self._IN_ATTRIB
            )
            watch_descriptor = libc.inotify_add_watch(
                fd,
                os.fsencode(self._directory),
                mask,
            )
            if watch_descriptor < 0:
                os.close(fd)
                return
        except Exception:
            return
        self._libc = libc
        self._inotify_fd = fd
        self._watch_descriptor = watch_descriptor


class LocalStageConsole:
    _REDRAW_INTERVAL_SECONDS = 0.1
    _MAX_BUFFERED_LINES = 2000

    def __init__(
        self,
        *,
        job_id: str,
        job_name: str,
        rundir: str | None,
        stage_index: int,
        total_stages: int,
        stage_workers: int,
        tracking_url: str | None,
    ) -> None:
        self._interactive = stdout_is_interactive()
        self._job_id = job_id
        self._job_name = job_name
        self._rundir = rundir
        self._stage_index = stage_index
        self._total_stages = total_stages
        self._tracking_url = tracking_url
        self._status = "running"
        self._worker_total = stage_workers
        self._worker_running = stage_workers
        self._worker_completed = 0
        self._worker_failed = 0
        self._elapsed_seconds = 0.0
        self._lines: deque[str] = deque(maxlen=self._MAX_BUFFERED_LINES)
        self._last_system_message: str | None = None
        self._last_rendered_at = 0.0
        self._last_rendered_line_count = 0
        self._alternate_screen = False
        self._cursor_hidden = False
        if self._interactive:
            self._write("\x1b[?1049h")
            self._alternate_screen = True
            self._write("\x1b[?25l")
            self._cursor_hidden = True
            self._render(force=True)

    def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
        formatted_lines = [
            self._format_line(worker_id=worker_id, line=line) for line in lines
        ]
        if not self._interactive:
            for line in formatted_lines:
                self._write(line + "\n")
            return
        for line in formatted_lines:
            if len(self._lines) == self._MAX_BUFFERED_LINES:
                self._render(force=True)
            self._lines.append(line)

    def set_status(self, status: str) -> None:
        self._status = status
        self._render(force=True)

    def emit_system(self, message: str) -> None:
        self._last_system_message = message
        self._lines.append(
            f"launcher: {message}"
            if not self._interactive
            else f"{_LABEL_COLOR}launcher:{_ANSI_RESET} {_VALUE_COLOR}{message}{_ANSI_RESET}"
        )
        self._render(force=True)

    def apply_snapshot(self, snapshot: LocalStageSnapshot) -> None:
        previous_stage_index = self._stage_index
        previous_total_stages = self._total_stages
        previous_status = self._status
        previous_total = self._worker_total
        previous_running = self._worker_running
        previous_completed = self._worker_completed
        previous_failed = self._worker_failed
        previous_elapsed_seconds = int(self._elapsed_seconds)
        self._stage_index = snapshot.stage_index
        self._total_stages = snapshot.total_stages
        self._status = snapshot.status
        self._worker_total = snapshot.worker_total
        self._worker_running = snapshot.worker_running
        self._worker_completed = snapshot.worker_completed
        self._worker_failed = snapshot.worker_failed
        self._elapsed_seconds = snapshot.elapsed_seconds
        if (
            self._stage_index != previous_stage_index
            or self._total_stages != previous_total_stages
            or self._status != previous_status
            or self._worker_total != previous_total
            or self._worker_running != previous_running
            or self._worker_completed != previous_completed
            or self._worker_failed != previous_failed
            or int(self._elapsed_seconds) != previous_elapsed_seconds
        ):
            self._render()

    def close(self) -> None:
        try:
            self._render(force=True)
            if self._interactive and self._cursor_hidden:
                if self._last_rendered_line_count > 0:
                    self._write(f"\x1b[{self._last_rendered_line_count};1H")
                self._write("\x1b[?25h")
            if self._alternate_screen:
                self._write("\x1b[?1049l")
                terminal_width = shutil.get_terminal_size(fallback=(120, 30)).columns
                self._write("\n".join(self._build_header_lines(width=terminal_width)))
                self._write("\n")
                if self._last_system_message is not None:
                    self._write(f"launcher: {self._last_system_message}\n")
            else:
                self._write("\n")
        except BrokenPipeError:
            pass
        finally:
            self._cursor_hidden = False
            self._alternate_screen = False

    def _build_header_lines(self, *, width: int) -> list[str]:
        title = "Macrodata Refiner".center(width)
        separator = f"{_SEPARATOR_COLOR}{'-' * width}{_ANSI_RESET}"
        left_width = max(20, (width - 3) // 2)
        right_width = max(20, width - left_width - 3)
        row_specs = [
            (
                "Job",
                self._job_name,
                _VALUE_COLOR,
                "Stage",
                self._format_stage_badges(
                    max_width=max(1, right_width - len("Stage") - 2)
                ),
                "",
            ),
            (
                "Job ID",
                self._job_id,
                _VALUE_COLOR,
                "Workers",
                self._format_worker_counts(
                    max_width=max(1, right_width - len("Workers") - 2)
                ),
                "",
            ),
            (
                "URL" if self._tracking_url is not None else "",
                self._tracking_url or "",
                _URL_COLOR,
                "Status",
                self._status,
                _STATUS_COLORS.get(self._status, _VALUE_COLOR),
            ),
        ]
        row_specs.insert(
            2,
            (
                "Rundir" if self._rundir is not None else "Runtime",
                self._rundir
                if self._rundir is not None
                else _format_elapsed_seconds(self._elapsed_seconds),
                _VALUE_COLOR,
                "Runtime" if self._rundir is not None else "",
                _format_elapsed_seconds(self._elapsed_seconds)
                if self._rundir is not None
                else "",
                "",
            ),
        )

        def build_row(
            left_label: str,
            left_value: str,
            right_label: str,
            right_value: str,
            left_value_color: str = _VALUE_COLOR,
            right_color: str = _VALUE_COLOR,
        ) -> str:
            left_plain = f"{left_label}: {left_value}" if left_label else left_value
            left_value_width = (
                max(1, left_width - len(left_label) - 2) if left_label else left_width
            )
            right_value_width = (
                max(1, right_width - len(right_label) - 2)
                if right_label
                else right_width
            )
            left_segment = (
                f"{left_value_color}{_truncate_plain(left_value, left_value_width)}{_ANSI_RESET}"
                if not left_label and left_value
                else (
                    f"{_LABEL_COLOR}{left_label}:{_ANSI_RESET} "
                    f"{left_value_color}{_truncate_plain(left_value, left_value_width)}{_ANSI_RESET}"
                    if left_label
                    else ""
                )
            )
            if not right_label:
                right_segment = ""
            else:
                right_segment = (
                    f"{_LABEL_COLOR}{right_label}:{_ANSI_RESET} "
                    f"{right_color}{_truncate_plain(right_value, right_value_width)}{_ANSI_RESET}"
                )
            return (
                (
                    _pad_right(left_segment, min(left_width, len(left_plain)))
                    if not self._interactive
                    else _pad_right(left_segment, left_width)
                )
                + " | "
                + _pad_right(right_segment, right_width)
            )

        rows: list[str] = [f"{_TITLE_COLOR}{title}{_ANSI_RESET}", separator]
        rows.extend(
            build_row(
                left_label,
                left_value,
                right_label,
                right_value,
                left_value_color,
                right_color,
            )
            for left_label, left_value, left_value_color, right_label, right_value, right_color in row_specs
        )
        rows.append(separator)
        return rows

    def _format_stage_badges(self, *, max_width: int) -> str:
        def stage_token(stage_number: int) -> str:
            label = str(stage_number)
            if stage_number == self._stage_index:
                color = _STATUS_COLORS.get(self._status, _VALUE_COLOR)
                return f"{color}[{label}]{_ANSI_RESET}"
            if self._interactive:
                return f"{_LABEL_COLOR}{label}{_ANSI_RESET}"
            return label

        ellipsis = f"{_SEPARATOR_COLOR}...{_ANSI_RESET}" if self._interactive else "..."
        badges: list[str] = []
        for stage_number in range(self._total_stages):
            badges.append(stage_token(stage_number))
        full = " ".join(badges)
        if _visible_width(full) <= max_width:
            return full

        compact: list[str] = []
        if self._stage_index > 0:
            compact.append(stage_token(0))
        if self._stage_index > 1:
            compact.append(ellipsis)
        compact.append(stage_token(self._stage_index))
        if self._stage_index < self._total_stages - 2:
            compact.append(ellipsis)
        if self._stage_index < self._total_stages - 1:
            compact.append(stage_token(self._total_stages - 1))
        compact_text = " ".join(compact)
        if _visible_width(compact_text) <= max_width:
            return compact_text
        return stage_token(self._stage_index)

    def _format_worker_counts(self, *, max_width: int) -> str:
        if not self._interactive:
            return _truncate_plain(
                f"running={self._worker_running} "
                f"completed={self._worker_completed} "
                f"failed={self._worker_failed} "
                f"total={self._worker_total}",
                max_width,
            )
        variants = [
            ("running", "completed", "failed", "total"),
            ("run", "done", "fail", "tot"),
            ("r", "d", "f", "t"),
        ]
        for variant in variants:
            text = " ".join(
                [
                    f"{variant[0]}={_STATUS_COLORS['running']}{self._worker_running}{_ANSI_RESET}",
                    f"{variant[1]}={_STATUS_COLORS['completed']}{self._worker_completed}{_ANSI_RESET}",
                    f"{variant[2]}={_STATUS_COLORS['failed']}{self._worker_failed}{_ANSI_RESET}",
                    f"{variant[3]}={_VALUE_COLOR}{self._worker_total}{_ANSI_RESET}",
                ]
            )
            if _visible_width(text) <= max_width:
                return text
        last = variants[-1]
        return " ".join(
            [
                f"{last[0]}={_STATUS_COLORS['running']}{self._worker_running}{_ANSI_RESET}",
                f"{last[1]}={_STATUS_COLORS['completed']}{self._worker_completed}{_ANSI_RESET}",
                f"{last[2]}={_STATUS_COLORS['failed']}{self._worker_failed}{_ANSI_RESET}",
                f"{last[3]}={_VALUE_COLOR}{self._worker_total}{_ANSI_RESET}",
            ]
        )

    def _format_line(self, *, worker_id: str, line: str) -> str:
        prefix = f"worker={worker_id}"
        if not self._interactive:
            return f"{prefix} {line}"
        worker_color = _WORKER_COLORS[
            sum(worker_id.encode("utf-8")) % len(_WORKER_COLORS)
        ]
        match = _LOGURU_LINE_RE.match(line)
        if match is None:
            return f"{worker_color}{prefix} {line}{_ANSI_RESET}"
        level = match.group("level")
        return (
            f"{worker_color}{prefix}{_ANSI_RESET} "
            f"{_TIMESTAMP_COLOR}{match.group('timestamp').strip()}{_ANSI_RESET} | "
            f"{_loguru_markup_to_ansi(logger.level(level).color) or worker_color}"
            f"{level}{match.group('level_padding')}{_ANSI_RESET} | "
            f"{worker_color}{match.group('rest')}{_ANSI_RESET}"
        )

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_rendered_at < self._REDRAW_INTERVAL_SECONDS:
            return
        self._last_rendered_at = now
        if not self._interactive:
            while self._lines:
                self._write(self._lines.popleft() + "\n")
            return
        terminal_size = shutil.get_terminal_size(fallback=(120, 30))
        header_lines = self._build_header_lines(width=terminal_size.columns)
        available_log_lines = max(1, terminal_size.lines - len(header_lines) - 1)
        visible_lines = list(self._lines)[-available_log_lines:]
        self._last_rendered_line_count = len(header_lines) + available_log_lines
        self._write("\x1b[H\x1b[J")
        self._write(
            "\n".join(
                [*header_lines, *visible_lines]
                + [""] * max(0, available_log_lines - len(visible_lines))
            )
        )

    @staticmethod
    def _write(text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError:
            try:
                devnull = open(os.devnull, "w", encoding="utf-8")
            except OSError:
                raise
            try:
                os.dup2(devnull.fileno(), sys.stdout.fileno())
            except (AttributeError, OSError, io.UnsupportedOperation):
                pass
            sys.stdout = devnull
            raise


def _drain_stream(
    stream: io.TextIOWrapper | None,
    *,
    target: list[str],
) -> None:
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            target.append(line)
    finally:
        stream.close()


def stream_local_stage_logs(
    *,
    console: LocalStageConsole,
    worker_log_paths: dict[str, Path],
    is_stage_running: Callable[[], bool],
    on_tick: Callable[[], None] | None = None,
    log_mode: str = "all",
    poll_interval_seconds: float = 0.05,
) -> None:
    error_continuation_open: dict[str, bool] = {}

    def _filter_lines(worker_id: str, lines: list[str]) -> list[str]:
        if log_mode != "errors":
            return [
                line
                for line in lines
                if should_emit_worker_line(
                    log_mode=log_mode,
                    worker_id=worker_id,
                    selected_worker_id=selected_worker_id,
                    line=line,
                )
            ]
        emitted: list[str] = []
        for line in lines:
            match = _LOGURU_LINE_RE.match(line)
            if match is None:
                if error_continuation_open.get(worker_id, False):
                    emitted.append(line)
                continue
            error_continuation_open[worker_id] = match.group("level").upper() in {
                "ERROR",
                "CRITICAL",
            }
            if error_continuation_open[worker_id]:
                emitted.append(line)
        return emitted

    if log_mode == "none":
        snapshot_interval_seconds = max(poll_interval_seconds, 0.25)
        next_snapshot_at = time.monotonic()
        while True:
            now = time.monotonic()
            if on_tick is not None and now >= next_snapshot_at:
                on_tick()
                next_snapshot_at = now + snapshot_interval_seconds
            if not is_stage_running():
                break
            time.sleep(max(0.0, next_snapshot_at - time.monotonic()))
        console._render()
        return
    selected_worker_id = (
        sorted(worker_log_paths)[0] if log_mode == "one" and worker_log_paths else None
    )
    tailed_worker_ids = (
        [selected_worker_id]
        if log_mode == "one" and selected_worker_id is not None
        else list(worker_log_paths)
    )
    log_tails = {
        worker_id: LocalStageLogTail(path=worker_log_paths[worker_id])
        for worker_id in tailed_worker_ids
    }
    watcher = LocalStageLogWatcher(
        paths={
            worker_id: worker_log_paths[worker_id] for worker_id in tailed_worker_ids
        }
    )
    snapshot_interval_seconds = max(poll_interval_seconds, 0.25)
    next_snapshot_at = time.monotonic()
    try:
        while True:
            emitted_any = False
            for worker_id, log_tail in log_tails.items():
                lines = _filter_lines(worker_id, log_tail.poll())
                if lines:
                    console.emit_lines(worker_id=worker_id, lines=lines)
                    emitted_any = True
            if emitted_any:
                console._render()
            now = time.monotonic()
            if on_tick is not None and now >= next_snapshot_at:
                on_tick()
                next_snapshot_at = now + snapshot_interval_seconds
            if not is_stage_running():
                break
            watcher.wait(max(0.0, next_snapshot_at - time.monotonic()))
        for worker_id, log_tail in log_tails.items():
            lines = _filter_lines(worker_id, log_tail.flush())
            if lines:
                console.emit_lines(worker_id=worker_id, lines=lines)
        console._render()
    finally:
        watcher.close()
        for log_tail in log_tails.values():
            log_tail.close()


def run_local_stage_ui(
    *,
    worker_log_paths: dict[str, Path],
    snapshot_getter: Callable[[], LocalStageSnapshot],
    log_mode: str = "all",
    interrupt_message: str | None = None,
    poll_interval_seconds: float = 0.05,
) -> None:
    snapshot = snapshot_getter()
    console = LocalStageConsole(
        job_id=snapshot.job_id,
        job_name=snapshot.job_name,
        rundir=snapshot.rundir,
        stage_index=snapshot.stage_index,
        total_stages=snapshot.total_stages,
        stage_workers=snapshot.stage_workers,
        tracking_url=snapshot.tracking_url,
    )
    console.apply_snapshot(snapshot)

    def _update_snapshot() -> None:
        nonlocal snapshot
        snapshot = snapshot_getter()
        console.apply_snapshot(snapshot)

    try:
        stream_local_stage_logs(
            console=console,
            worker_log_paths=worker_log_paths,
            is_stage_running=lambda: snapshot.status == "running",
            on_tick=_update_snapshot,
            log_mode=normalize_log_mode(log_mode),
            poll_interval_seconds=poll_interval_seconds,
        )
        console.apply_snapshot(snapshot)
    except KeyboardInterrupt:
        console.set_status("failed")
        if stdout_is_interactive():
            console.emit_system(interrupt_message or "Local job interrupted.")
        raise
    except BrokenPipeError:
        raise
    except Exception as err:
        console.set_status("failed")
        console.emit_system(str(err))
        raise
    finally:
        console.close()


__all__ = [
    "LocalStageSnapshot",
    "LocalStageConsole",
    "LocalStageLogTail",
    "LocalLaunchInterrupted",
    "LocalLaunchResumeError",
    "LaunchStats",
    "format_resume_message",
    "run_local_stage_ui",
    "stdout_is_interactive",
    "stream_local_stage_logs",
]
