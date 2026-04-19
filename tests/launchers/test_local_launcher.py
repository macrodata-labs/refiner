from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
import io
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, cast
import pytest
from loguru import logger as _base_logger

import sys as local_run_sys
from refiner.cli.run.local import collect_local_stage_results
from refiner.cli.ui.console import (
    StageConsole,
    StageSnapshot,
    normalize_log_mode,
    run_stage_ui,
    should_emit_worker_line,
    stream_stage_logs,
)
from refiner.pipeline.data.shard import FilePart, Shard
from refiner.pipeline import RefinerPipeline, read_csv, read_jsonl
from refiner.launchers.local import LaunchStats, LocalLauncher
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.data.row import DictRow, Row
from refiner.platform.auth import MacrodataCredentialsError
from refiner.worker.resources.gpu import build_gpu_sets


@pytest.fixture(autouse=True)
def _disable_local_init_api_ping(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("refiner.launchers.local.request_json", lambda **kwargs: {})
    monkeypatch.setenv("REFINER_WORKDIR", str(tmp_path))


class _FakeReader(BaseReader):
    def __init__(
        self,
        shards: list[Shard],
        rows_by_shard_id: Mapping[str, Sequence[Row]],
    ):
        super().__init__(inputs=[])
        self._shards = shards
        self._rows_by_shard_id = rows_by_shard_id

    def list_shards(self) -> list[Shard]:
        return list(self._shards)

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        yield from self._rows_by_shard_id.get(shard.id, [])


def test_launch_local_single_worker(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n{"x": 2}\n')
    p2.write_text('{"x": 3}\n')

    pipeline = (
        read_jsonl([str(p1), str(p2)])
        .map(lambda r: {"x": int(r["x"]) + 1})
        .filter(lambda r: int(r["x"]) % 2 == 0)
    )

    stats = pipeline.launch_local(
        name="unit-test-local",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_launch_local_rejects_detach_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFINER_ATTACH", "detach")

    with pytest.raises(SystemExit, match="--detach is only supported"):
        read_jsonl("input.jsonl").launch_local(name="unit-test-local")


def test_launch_local_single_worker_csv(tmp_path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("x\n1\n2\n")

    pipeline = (
        read_csv(str(path))
        .map(lambda r: {"x": int(r["x"]) + 1})
        .filter(lambda r: int(r["x"]) % 2 == 0)
    )

    stats = pipeline.launch_local(
        name="unit-test-local-csv",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 1


def test_launch_local_coalesces_writer_shards(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')

    pipeline = read_jsonl([str(p1), str(p2)], num_shards=1).write_jsonl(
        tmp_path / "out"
    )

    stats = pipeline.launch_local(
        name="unit-test-local-coalesced",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.claimed == 2
    assert stats.completed == 2


def test_build_gpu_sets_partitions_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0", "1", "2", "3"],
    )
    sets = build_gpu_sets(num_workers=2, gpus_per_worker=2)
    assert sets == [["0", "1"], ["2", "3"]]


def test_build_gpu_sets_raises_when_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0"],
    )
    with pytest.raises(ValueError):
        build_gpu_sets(num_workers=2, gpus_per_worker=1)


def test_launch_local_assigns_visible_gpus(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    monkeypatch.setattr(
        "refiner.worker.resources.gpu.available_gpu_ids",
        lambda: ["0"],
    )

    stats = pipeline.launch_local(
        name="local-gpu-launch",
        num_workers=1,
        gpus_per_worker=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.completed == 1
    assert stats.failed == 0


def test_launch_local_multi_worker_subprocess_with_lambda(tmp_path) -> None:
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text('{"x": 1}\n')
    p2.write_text('{"x": 2}\n')
    pipeline = read_jsonl([str(p1), str(p2)]).map(lambda r: {"x": int(r["x"]) + 10})

    stats = pipeline.launch_local(
        name="unit-test-local-subprocess",
        num_workers=2,
        rundir=str(tmp_path / "run"),
    )
    assert stats.workers == 2
    assert stats.claimed == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 2


def test_launch_local_ignores_non_json_stdout_before_final_stats(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')

    def noisy_map(row):
        print(f"processing {row['x']}")
        return {"x": int(row["x"]) + 1}

    pipeline = read_jsonl(str(path)).map(noisy_map)

    stats = pipeline.launch_local(
        name="unit-test-local-noisy-stdout",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.workers == 1
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.output_rows == 1


def test_launch_local_writes_worker_loguru_logs_to_stage_log_files(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')

    def logged_map(row):
        from refiner import logger

        logger.info("processing row {}", row["x"])
        return {"x": int(row["x"]) + 1}

    pipeline = read_jsonl(str(path)).map(logged_map)
    rundir = tmp_path / "run"

    stats = pipeline.launch_local(
        name="unit-test-local-loguru-file",
        num_workers=1,
        rundir=str(rundir),
    )

    assert stats.completed == 1
    log_files = list((rundir / "stage-0" / "logs").glob("worker-*.log"))
    assert len(log_files) == 1
    assert "processing row 1" in log_files[0].read_text()


def test_launch_local_streams_worker_logs_to_launcher_stdout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')

    def logged_map(row):
        from refiner import logger

        logger.info("processing row {}", row["x"])
        return {"x": int(row["x"]) + 1}

    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)

    pipeline = read_jsonl(str(path)).map(logged_map)
    pipeline.launch_local(
        name="unit-test-local-log-stream",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    out = capsys.readouterr().out
    assert "worker=" in out
    assert "processing row 1" in out


def test_local_stage_console_colors_timestamp_level_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    console = StageConsole(
        job_id="job-1",
        job_name="local-log-stream-demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=3,
        stage_workers=1,
        tracking_url=None,
    )
    try:
        header_lines = console._build_header_lines(width=80)
        line = console._format_line(
            worker_id="worker-1",
            line=(
                "2026-04-16 21:00:21.014 | INFO     | "
                "__main__:emit_logs:14 - loguru row=0 starting"
            ),
        )
    finally:
        console.close()

    assert "Macrodata Refiner" in header_lines[0]
    assert "Job:" in header_lines[2]
    assert "local-log-stream-demo" in header_lines[2]
    assert "Stage:" in header_lines[2]
    assert "[0]" in header_lines[2]
    assert "1" in header_lines[2]
    assert "2" in header_lines[2]
    assert "Job ID:" in header_lines[3]
    assert "Workers:" in header_lines[3]
    assert "running=\x1b[" in header_lines[3] or "run=\x1b[" in header_lines[3]
    assert "\x1b[1;38;5;220m1\x1b[0m" in header_lines[3]
    assert "Rundir:" in header_lines[4]
    assert "Runtime:" in header_lines[4]
    assert "00:00" in header_lines[4]
    assert "Status:" in header_lines[5]
    assert "running" in header_lines[5]
    assert "worker=worker-1" in line
    assert "\x1b[38;5;255m2026-04-16 21:00:21.014\x1b[0m" in line
    info_markup = _base_logger.level("INFO").color
    expected_info_prefix = info_markup.replace("<bold>", "\x1b[1m").replace(
        "<green>", "\x1b[32m"
    )
    assert f"{expected_info_prefix}INFO" in line
    assert "\x1b[0m | " in line
    assert "__main__:emit_logs:14 - loguru row=0 starting\x1b[0m" in line


def test_local_stage_console_omits_rundir_row_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    console = StageConsole(
        job_id="job-1",
        job_name="cloud-attach-demo",
        rundir=None,
        stage_index=0,
        total_stages=2,
        stage_workers=1,
        tracking_url="https://example.com/jobs/job-1",
    )
    try:
        header_lines = console._build_header_lines(width=80)
    finally:
        console.close()

    assert not any("Rundir:" in line for line in header_lines)
    assert any("Runtime:" in line for line in header_lines)


def test_local_stage_console_apply_snapshot_updates_stage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)
    console = StageConsole(
        job_id="job-1",
        job_name="demo",
        rundir=None,
        stage_index=0,
        total_stages=2,
        stage_workers=1,
        tracking_url=None,
    )
    try:
        console.apply_snapshot(
            StageSnapshot(
                job_id="job-1",
                job_name="demo",
                rundir=None,
                stage_index=1,
                total_stages=3,
                stage_workers=4,
                tracking_url=None,
                status="running",
                worker_total=4,
                worker_running=2,
                worker_completed=1,
                worker_failed=0,
                elapsed_seconds=5.0,
            )
        )
    finally:
        console.close()

    assert console._stage_index == 1
    assert console._total_stages == 3


def test_local_stage_console_formats_system_lines_without_worker_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole._render",
        lambda *args, **kwargs: None,
    )
    console = StageConsole(
        job_id="job-1",
        job_name="local-log-stream-demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
    )
    try:
        console.emit_system("local run interrupted; shutting down workers")
        line = console._lines[-1]
    finally:
        console.close()

    assert "launcher:" in line
    assert "worker=launcher" not in line


def test_local_stage_console_bounds_interactive_log_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole._render",
        lambda *args, **kwargs: None,
    )
    console = StageConsole(
        job_id="job-1",
        job_name="local-log-stream-demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
    )
    try:
        for index in range(StageConsole._MAX_BUFFERED_LINES + 25):
            console.emit_lines(worker_id="worker-1", lines=[f"line {index}"])
        assert len(console._lines) == StageConsole._MAX_BUFFERED_LINES
        assert any("line 24" in line for line in console._lines)
        assert not any("line 0" in line for line in console._lines)
    finally:
        console.close()


def test_local_stage_console_close_moves_cursor_below_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole._write",
        staticmethod(lambda text: writes.append(text)),
    )

    console = StageConsole(
        job_id="job-1",
        job_name="local-log-stream-demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
    )
    console.close()

    assert any(text.startswith("\x1b[") and text.endswith(";1H") for text in writes)
    assert "\x1b[?25h" in writes
    assert "\x1b[?1049l" in writes
    assert any("Macrodata Refiner" in text for text in writes)
    assert writes[-1] == "\n"


def test_local_stage_console_close_prints_last_system_message_after_alt_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole._write",
        staticmethod(lambda text: writes.append(text)),
    )

    console = StageConsole(
        job_id="job-1",
        job_name="local-log-stream-demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
    )
    console.emit_system("Local job interrupted.")
    console.close()

    assert any("launcher: Local job interrupted." in text for text in writes)


def test_local_stage_console_write_squelches_broken_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenStdout:
        def write(self, text: str) -> None:
            raise BrokenPipeError()

        def flush(self) -> None:
            return None

        def fileno(self) -> int:
            raise io.UnsupportedOperation()

    broken = _BrokenStdout()
    monkeypatch.setattr("refiner.cli.ui.console.sys.stdout", broken)

    with pytest.raises(BrokenPipeError):
        StageConsole._write("hello")

    assert local_run_sys.stdout is not broken


def test_run_stage_ui_marks_failed_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []
    system_messages: list[str] = []
    closed: list[bool] = []

    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)
    monkeypatch.setattr(
        "refiner.cli.ui.console.stream_stage_logs",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole.set_status",
        lambda self, status: statuses.append(status),
    )
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole.emit_system",
        lambda self, message: system_messages.append(message),
    )
    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole.close",
        lambda self: closed.append(True),
    )

    snapshot = StageSnapshot(
        job_id="job-1",
        job_name="demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
        status="running",
        worker_total=1,
        worker_running=1,
        worker_completed=0,
        worker_failed=0,
        elapsed_seconds=0.0,
    )

    with pytest.raises(KeyboardInterrupt):
        run_stage_ui(
            worker_log_paths={},
            snapshot_getter=lambda: snapshot,
        )

    assert statuses[-1] == "failed"
    assert system_messages == []
    assert closed == [True]


def test_collect_local_stage_results_marks_failed_from_worker_payload_and_emits_log_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeConsole:
        def __init__(self) -> None:
            self.snapshots: list[StageSnapshot] = []
            self.system_messages: list[str] = []
            self.lines: list[tuple[str, list[str]]] = []
            self.closed = 0

        def apply_snapshot(self, snapshot: StageSnapshot) -> None:
            self.snapshots.append(snapshot)

        def emit_system(self, message: str) -> None:
            self.system_messages.append(message)

        def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
            self.lines.append((worker_id, list(lines)))

        def close(self) -> None:
            self.closed += 1

    stage_dir = tmp_path / "stage-0"
    logs_dir = stage_dir / "logs"
    logs_dir.mkdir(parents=True)
    worker_id = "worker-1"
    log_path = logs_dir / f"worker-{worker_id}.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-04-19 19:00:00.000 | INFO     | demo:start - starting",
                "Traceback (most recent call last):",
                '  File "demo.py", line 1, in <module>',
                "ValueError: boom",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "print(json.dumps({"
                "'worker_id':'worker-1','claimed':1,'completed':0,'failed':1,"
                "'output_rows':0,'error':'boom'}), flush=True)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fake_console = _FakeConsole()
    monkeypatch.setattr(
        "refiner.cli.run.local.run_stage_ui",
        lambda **kwargs: (
            fake_console,
            StageSnapshot(
                job_id="job-1",
                job_name="demo",
                rundir=str(tmp_path),
                stage_index=0,
                total_stages=1,
                stage_workers=1,
                tracking_url=None,
                status="completed",
                worker_total=1,
                worker_running=0,
                worker_completed=1,
                worker_failed=0,
                elapsed_seconds=0.0,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="worker worker-1: boom"):
        collect_local_stage_results(
            job_id="job-1",
            job_name="demo",
            rundir=str(tmp_path),
            stage_index=0,
            total_stages=1,
            stage_workers=1,
            tracking_url=None,
            processes=[(worker_id, process)],
            log_mode=None,
            interrupt_message="interrupted",
            terminate_timeout_seconds=0.1,
        )

    assert fake_console.snapshots[-1].status == "failed"
    assert fake_console.snapshots[-1].worker_failed == 1
    assert fake_console.snapshots[-1].worker_completed == 0
    assert any(
        "last log lines from worker-1:" == message
        for message in fake_console.system_messages
    )
    assert fake_console.lines[-1][0] == "worker-1"
    assert "ValueError: boom" in fake_console.lines[-1][1]
    assert fake_console.closed == 1


def test_stream_stage_logs_skips_tails_for_none_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)

    class _Console:
        def emit(self, *, worker_id: str, line: str) -> None:
            raise AssertionError(f"unexpected emit from {worker_id}: {line}")

        def _render(self, *, force: bool = False) -> None:
            del force
            return None

    monkeypatch.setattr(
        "refiner.cli.ui.console.StageLogTail",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tail should not be created")
        ),
    )
    ticks = 0

    def _is_stage_running() -> bool:
        nonlocal ticks
        ticks += 1
        return ticks == 1

    stream_stage_logs(
        console=cast(Any, _Console()),
        worker_log_paths={"worker-1": tmp_path / "worker-1.log"},
        is_stage_running=_is_stage_running,
        log_mode="none",
        poll_interval_seconds=0.0,
    )


def test_stream_stage_logs_only_tails_selected_worker_for_one_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)
    tailed: list[Path] = []

    class _FakeTail:
        def __init__(self, *, path: Path) -> None:
            tailed.append(path)

        def poll(self) -> list[str]:
            return []

        def flush(self) -> list[str]:
            return []

        def close(self) -> None:
            return None

    class _Console:
        def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
            raise AssertionError(f"unexpected emit from {worker_id}: {lines}")

        def _render(self, *, force: bool = False) -> None:
            del force
            return None

    monkeypatch.setattr("refiner.cli.ui.console.StageLogTail", _FakeTail)
    ticks = 0

    def _is_stage_running() -> bool:
        nonlocal ticks
        ticks += 1
        return ticks == 1

    stream_stage_logs(
        console=cast(Any, _Console()),
        worker_log_paths={
            "worker-b": tmp_path / "worker-b.log",
            "worker-a": tmp_path / "worker-a.log",
            "worker-c": tmp_path / "worker-c.log",
        },
        is_stage_running=_is_stage_running,
        log_mode="one",
        poll_interval_seconds=0.0,
    )

    assert tailed == [tmp_path / "worker-a.log"]


def test_local_stage_console_flushes_before_batched_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renders: list[bool] = []

    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: True)
    original_render = StageConsole._render

    def _recording_render(self, *, force: bool = False) -> None:
        renders.append(force)

    monkeypatch.setattr(
        "refiner.cli.ui.console.StageConsole._render",
        _recording_render,
    )

    console = StageConsole(
        job_id="job-1",
        job_name="demo",
        rundir="/tmp/run",
        stage_index=0,
        total_stages=1,
        stage_workers=1,
        tracking_url=None,
    )
    try:
        console._lines = deque(
            [f"line {index}" for index in range(StageConsole._MAX_BUFFERED_LINES)],
            maxlen=StageConsole._MAX_BUFFERED_LINES,
        )
        console.emit_lines(
            worker_id="worker-1",
            lines=["overflow 1", "overflow 2"],
        )
    finally:
        monkeypatch.setattr(
            "refiner.cli.ui.console.StageConsole._render",
            original_render,
        )
        console.close()

    assert True in renders


def test_stream_stage_logs_errors_mode_keeps_traceback_continuations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("refiner.cli.ui.console.stdout_is_interactive", lambda: False)
    emitted: list[str] = []

    class _FakeTail:
        def __init__(self, *, path: Path) -> None:
            del path
            self._emitted = False

        def poll(self) -> list[str]:
            if self._emitted:
                return []
            self._emitted = True
            return [
                "2026-04-16 21:00:21.014 | ERROR    | worker:run - failed",
                "Traceback (most recent call last):",
                '  File "demo.py", line 1, in <module>',
                "ValueError: boom",
                "2026-04-16 21:00:21.015 | INFO     | worker:run - ignored",
            ]

        def flush(self) -> list[str]:
            return []

        def close(self) -> None:
            return None

    class _Console:
        def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
            del worker_id
            emitted.extend(lines)

        def _render(self, *, force: bool = False) -> None:
            del force
            return None

    monkeypatch.setattr("refiner.cli.ui.console.StageLogTail", _FakeTail)
    ticks = 0

    def _is_stage_running() -> bool:
        nonlocal ticks
        ticks += 1
        return ticks == 1

    stream_stage_logs(
        console=cast(Any, _Console()),
        worker_log_paths={"worker-a": tmp_path / "worker-a.log"},
        is_stage_running=_is_stage_running,
        log_mode="errors",
        poll_interval_seconds=0.0,
    )

    assert emitted == [
        "2026-04-16 21:00:21.014 | ERROR    | worker:run - failed",
        "Traceback (most recent call last):",
        '  File "demo.py", line 1, in <module>',
        "ValueError: boom",
    ]


def test_normalize_log_mode_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="unsupported log mode"):
        normalize_log_mode("bad")


def test_should_emit_worker_line_filters_by_mode() -> None:
    info_line = "2026-04-16 21:00:21.014 | INFO     | worker:run - started"
    error_line = "2026-04-16 21:00:21.014 | ERROR    | worker:run - failed"

    assert should_emit_worker_line(
        log_mode="all",
        worker_id="worker-a",
        selected_worker_id=None,
        line=info_line,
    )
    assert not should_emit_worker_line(
        log_mode="none",
        worker_id="worker-a",
        selected_worker_id=None,
        line=info_line,
    )
    assert should_emit_worker_line(
        log_mode="one",
        worker_id="worker-a",
        selected_worker_id="worker-a",
        line=info_line,
    )
    assert not should_emit_worker_line(
        log_mode="one",
        worker_id="worker-b",
        selected_worker_id="worker-a",
        line=info_line,
    )
    assert should_emit_worker_line(
        log_mode="errors",
        worker_id="worker-a",
        selected_worker_id=None,
        line=error_line,
    )
    assert not should_emit_worker_line(
        log_mode="errors",
        worker_id="worker-a",
        selected_worker_id=None,
        line=info_line,
    )
    assert should_emit_worker_line(
        log_mode="errors",
        worker_id="worker-a",
        selected_worker_id=None,
        line=info_line,
        severity="error",
    )
    assert not should_emit_worker_line(
        log_mode="errors",
        worker_id="worker-a",
        selected_worker_id=None,
        line=error_line,
        severity="info",
    )


def test_local_launcher_registers_job_and_reports_stage_lifecycle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    create_calls: list[dict[str, object]] = []
    started: list[tuple[str, int]] = []
    finished: list[tuple[str, int, str]] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            create_calls.append(kwargs)
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            started.append((job_id, stage_index))

        def report_stage_finished(
            self,
            *,
            job_id: str,
            stage_index: int,
            status: str,
            reason: str | None = None,
        ):
            assert reason is None
            finished.append((job_id, stage_index, status))

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )
    monkeypatch.setattr(
        LocalLauncher,
        "_build_local_job_id",
        staticmethod(lambda name: "job-local"),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-register",
        num_workers=1,
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda *, stage: LaunchStats(
            job_id="job-remote",
            workers=1,
            claimed=1,
            completed=1,
            failed=0,
            output_rows=1,
        ),
    )
    launcher.launch()

    assert create_calls
    assert create_calls[0]["executor"] == {"type": "refiner-local"}
    manifest = cast(dict[str, object], create_calls[0]["manifest"])
    environment = cast(dict[str, object], manifest["environment"])
    assert environment["rundir"] == str(tmp_path / "runs" / "<jobid>")
    assert started == [("job-remote", 0)]
    assert finished == [("job-remote", 0, "completed")]


def test_local_launcher_suppresses_startup_info_logs_in_interactive_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    info_messages: list[str] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr("refiner.launchers.local.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.launchers.local.logger.info",
        lambda *args, **kwargs: info_messages.append(str(args[0])),
    )
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )
    monkeypatch.setattr(
        LocalLauncher,
        "_build_local_job_id",
        staticmethod(lambda name: "job-local"),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-register",
        num_workers=1,
    )
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    assert info_messages == []


def test_local_launcher_prints_plain_status_lines_in_noninteractive_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr("refiner.launchers.local.stdout_is_interactive", lambda: False)
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )
    monkeypatch.setattr(
        LocalLauncher,
        "_build_local_job_id",
        staticmethod(lambda name: "job-local"),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-register",
        num_workers=1,
    )
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    err = capsys.readouterr().err
    assert "launcher: Local job registered. View job:" in err
    assert "launcher: Starting local job" in err
    assert "launcher: Local job completed" in err
    assert "refiner.launchers.local" not in err


def test_local_launcher_warns_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    shard = Shard.from_file_parts([FilePart(path="a", start=0, end=1)])
    rows = {shard.id: [DictRow({"x": 1})]}
    pipeline = RefinerPipeline(source=_FakeReader([shard], rows))
    warnings: list[str] = []

    monkeypatch.setattr(
        "refiner.launchers.local.current_api_key",
        lambda: (_ for _ in ()).throw(
            MacrodataCredentialsError("missing", missing=True)
        ),
    )
    monkeypatch.setattr(
        "refiner.launchers.local.logger.warning",
        lambda message: warnings.append(message),
    )
    monkeypatch.setattr(
        LocalLauncher,
        "_build_local_job_id",
        staticmethod(lambda name: "job-local"),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-no-creds",
        num_workers=1,
    )
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    assert warnings == [
        "No valid Macrodata API key found. Run `macrodata login` to track local jobs."
    ]
    assert launcher.job_id == "job-local"


def test_local_launcher_warns_for_invalid_credentials_and_continues_locally(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    warnings: list[str] = []

    class _FakeClient:
        def create_job(self, **kwargs):  # noqa: ANN003, ANN204
            raise MacrodataCredentialsError("Unauthorized", missing=False)

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_bad")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )
    monkeypatch.setattr(
        "refiner.launchers.local.logger.warning",
        lambda message: warnings.append(message),
    )
    monkeypatch.setattr(
        LocalLauncher,
        "_build_local_job_id",
        staticmethod(lambda name: "job-local"),
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-launch-invalid-creds",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    assert warnings == [
        "Your Macrodata API key is invalid. Run `macrodata login` or set MACRODATA_API_KEY with a valid key. Local execution will continue without job tracking."
    ]
    assert launcher.job_id == "job-local"


def test_local_launcher_normalizes_explicit_rundir_on_init(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    monkeypatch.chdir(tmp_path)

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-explicit-rundir-normalized",
        num_workers=1,
        rundir="custom-run",
    )

    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    launcher.launch()

    assert launcher.rundir == str((tmp_path / "custom-run").resolve())


def test_launch_local_runs_planned_stages_sequentially(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = tmp_path / "stage0.jsonl"
    second_path = tmp_path / "stage1.jsonl"
    first_path.write_text('{"x": 1}\n{"x": 2}\n')
    second_path.write_text('{"x": 3}\n')

    pipeline = read_jsonl(str(first_path))
    stage_zero = read_jsonl(str(first_path))
    stage_one = read_jsonl(str(second_path)).map(lambda row: {"x": int(row["x"]) + 10})

    monkeypatch.setattr(
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=stage_zero,
                compute=StageComputeRequirements(num_workers=1),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=stage_one,
                compute=StageComputeRequirements(num_workers=2),
            ),
        ],
    )

    rundir = tmp_path / "run"
    stats = pipeline.launch_local(
        name="unit-test-local-multi-stage",
        num_workers=4,
        rundir=str(rundir),
    )

    assert stats.workers == 3
    assert stats.claimed == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.output_rows == 3
    assert (rundir / "stage-0").exists()
    assert (rundir / "stage-1").exists()


def test_launch_local_uses_explicit_rundir(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    rundir = tmp_path / "custom-run"

    stats = pipeline.launch_local(
        name="unit-test-local-rundir",
        num_workers=1,
        rundir=str(rundir),
    )

    assert (rundir / "stage-0").exists()
    assert not (tmp_path / "runs" / stats.job_id).exists()


def test_launch_local_resumes_from_existing_rundir(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n{"x": 2}\n')
    pipeline = read_jsonl(str(path), num_shards=2)
    rundir = tmp_path / "resume-run"
    stage_manifest = rundir / "stage-0" / "completed" / "worker-1.jsonl"
    first_shard = pipeline.list_shards()[0]
    stage_manifest.parent.mkdir(parents=True, exist_ok=True)
    stage_manifest.write_text(f'{{"shard_id": "{first_shard.id}"}}\n')

    stats = pipeline.launch_local(
        name="unit-test-local-resume",
        num_workers=2,
        rundir=str(rundir),
    )

    assert stats.claimed == 1
    assert stats.completed == 1


def test_local_launcher_stops_after_failed_stage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = read_jsonl(str(tmp_path / "missing.jsonl"))
    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stop-on-failure",
        num_workers=1,
    )

    stages = [
        PlannedStage(
            index=0,
            name="stage_0",
            pipeline=pipeline,
            compute=StageComputeRequirements(num_workers=1),
        ),
        PlannedStage(
            index=1,
            name="stage_1",
            pipeline=pipeline,
            compute=StageComputeRequirements(num_workers=1),
        ),
    ]
    launched: list[int] = []

    monkeypatch.setattr(launcher, "_planned_stages", lambda: stages)

    def fake_launch_stage(*, stage):  # noqa: ANN001
        launched.append(stage.index)
        return LaunchStats(
            job_id="job-1",
            workers=1,
            claimed=1,
            completed=0,
            failed=1 if stage.index == 0 else 0,
            output_rows=0,
        )

    monkeypatch.setattr(launcher, "_launch_stage", fake_launch_stage)

    with pytest.raises(RuntimeError, match=r"stage 0 failed"):
        launcher.launch()

    assert launched == [0]


def test_local_launcher_does_not_force_platform_terminal_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-no-forced-platform-finish",
        num_workers=1,
    )

    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )

    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda *, stage: LaunchStats(
            job_id="job-1",
            workers=1,
            claimed=1,
            completed=1,
            failed=0,
            output_rows=1,
        ),
    )
    stats = launcher.launch()

    assert stats.completed == 1


def test_local_launcher_reports_failed_stage_to_tracking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    finished: list[tuple[str, int, str]] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            return None

        def report_stage_finished(
            self,
            *,
            job_id: str,
            stage_index: int,
            status: str,
            reason: str | None = None,
        ):
            assert reason is None
            finished.append((job_id, stage_index, status))

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stage-fail-report",
        num_workers=1,
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match=r"boom.*rundir="):
        launcher.launch()

    assert finished == [("job-remote", 0, "failed")]


def test_local_launcher_reports_interrupted_stage_with_reason(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    finished: list[dict[str, object]] = []
    heartbeat_started: list[int] = []

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            return None

        def report_stage_finished(self, **kwargs):
            finished.append(kwargs)

    class _DummyThread:
        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stage-interrupt-report",
        num_workers=1,
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_start_stage_heartbeat",
        lambda *, tracking_client, stage_index: (
            heartbeat_started.append(stage_index) or threading.Event(),
            _DummyThread(),
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        launcher.launch()

    assert heartbeat_started == [0]
    assert "Local job interrupted" in str(exc_info.value)
    assert "rundir=" in str(exc_info.value)
    assert finished == [
        {
            "job_id": "job-remote",
            "stage_index": 0,
            "status": "failed",
            "reason": "Local launcher interrupted",
        }
    ]


def test_local_launcher_prints_abort_message_in_noninteractive_mode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    monkeypatch.setattr("refiner.launchers.local.stdout_is_interactive", lambda: False)

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stage-interrupt-output",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        launcher,
        "_register_tracked_job",
        lambda *, stages: (None, None),
    )

    with pytest.raises(KeyboardInterrupt):
        launcher.launch()

    err = capsys.readouterr().err
    assert "launcher: Local job interrupted during stage 0." in err
    assert "launcher: Local job interrupted." in err


def test_local_launcher_resets_stale_tracking_url_between_runs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stale-url-reset",
        num_workers=1,
    )
    launcher.job_tracking_url = "https://macrodata.co/old-job"
    monkeypatch.setattr(launcher, "_planned_stages", lambda: [])
    monkeypatch.setattr(
        launcher, "_register_tracked_job", lambda *, stages: (None, None)
    )

    stats = launcher.launch()

    assert launcher.job_tracking_url is None
    assert stats.job_id.startswith("local-stale-url-reset-")


def test_local_launcher_does_not_fail_stage_when_heartbeat_delivery_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n')
    pipeline = read_jsonl(str(path))
    finished: list[dict[str, object]] = []
    heartbeat_attempts = 0

    class _FakeClient:
        base_url = "https://macrodata.co"

        def create_job(self, **kwargs):
            from refiner.platform.client.models import CreateJobResponse

            return CreateJobResponse(
                job_id="job-remote",
                stage_index=0,
                workspace_slug="workspace",
            )

        def report_stage_started(self, *, job_id: str, stage_index: int):
            return None

        def report_stage_heartbeat(self, *, job_id: str, stage_index: int):
            nonlocal heartbeat_attempts
            heartbeat_attempts += 1
            raise RuntimeError("heartbeat offline")

        def report_stage_finished(self, **kwargs):
            finished.append(kwargs)

    monkeypatch.setattr("refiner.launchers.local.current_api_key", lambda: "md_test")
    monkeypatch.setattr(
        "refiner.launchers.local.MacrodataClient", lambda **kwargs: _FakeClient()
    )

    launcher = LocalLauncher(
        pipeline=pipeline,
        name="local-stage-heartbeat-delivery-failure",
        num_workers=1,
    )
    launcher._STAGE_HEARTBEAT_INTERVAL_SECONDS = 0.01
    launcher._STAGE_HEARTBEAT_FAILURE_THRESHOLD = 2
    monkeypatch.setattr(
        launcher,
        "_planned_stages",
        lambda: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )

    def fake_launch_stage(*, stage):  # noqa: ANN001
        time.sleep(0.05)
        return LaunchStats(
            job_id="job-remote",
            workers=1,
            claimed=1,
            completed=1,
            failed=0,
            output_rows=1,
        )

    monkeypatch.setattr(
        launcher,
        "_launch_stage",
        fake_launch_stage,
    )

    stats = launcher.launch()

    assert stats.completed == 1
    assert heartbeat_attempts >= 2
    assert finished == [
        {
            "job_id": "job-remote",
            "stage_index": 0,
            "status": "completed",
        }
    ]
