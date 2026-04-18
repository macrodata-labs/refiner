from __future__ import annotations

from refiner.cli.main import build_parser, main


def test_parser_has_auth_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["whoami"])
    assert args.command == "whoami"


def test_parser_has_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--logs", "one", "script.py", "--", "--rows", "10"]
    )
    assert args.command == "run"
    assert args.logs == "one"
    assert args.script == "script.py"
    assert args.script_args == ["--rows", "10"]


def test_parser_has_jobs_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["jobs", "list", "--kind", "cloud", "--me"])
    assert args.command == "jobs"
    assert args.jobs_command == "list"
    assert args.kind == "cloud"
    assert args.me is True


def test_parser_has_jobs_logs_follow_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["jobs", "logs", "job-1", "--follow"])
    assert args.command == "jobs"
    assert args.jobs_command == "logs"
    assert args.job_id == "job-1"
    assert args.follow is True


def test_parser_has_jobs_logs_cursor_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["jobs", "logs", "job-1", "--cursor", "cursor-1"])
    assert args.command == "jobs"
    assert args.jobs_command == "logs"
    assert args.job_id == "job-1"
    assert args.cursor == "cursor-1"


def test_parser_has_jobs_cancel_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["jobs", "cancel", "job-1"])
    assert args.command == "jobs"
    assert args.jobs_command == "cancel"
    assert args.job_id == "job-1"


def test_parser_has_jobs_workers_pagination_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["jobs", "workers", "job-1", "--limit", "50", "--cursor", "20"]
    )
    assert args.command == "jobs"
    assert args.jobs_command == "workers"
    assert args.job_id == "job-1"
    assert args.limit == 50
    assert args.cursor == "20"


def test_parser_has_stage_metrics_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["jobs", "metrics", "job-1", "2", "--step", "3", "--metric", "rows"]
    )
    assert args.command == "jobs"
    assert args.jobs_command == "metrics"
    assert args.job_id == "job-1"
    assert args.stage_index == 2
    assert args.step == 3
    assert args.metric == ["rows"]


def test_parser_has_resource_metrics_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["jobs", "resource-metrics", "job-1", "2", "--worker-id", "worker-1"]
    )
    assert args.command == "jobs"
    assert args.jobs_command == "resource-metrics"
    assert args.job_id == "job-1"
    assert args.stage_index == 2
    assert args.worker_id == ["worker-1"]


def test_main_dispatches(monkeypatch) -> None:
    monkeypatch.setattr("refiner.cli.main.cmd_whoami", lambda args: 7)
    rc = main(["whoami"])
    assert rc == 7


def test_main_no_args_shows_help(capsys) -> None:
    rc = main([])
    out = capsys.readouterr()
    assert rc == 0
    assert "Macrodata CLI" in out.out


def test_main_jobs_without_subcommand_shows_jobs_help(capsys) -> None:
    rc = main(["jobs"])
    out = capsys.readouterr()
    assert rc == 0
    assert "usage: macrodata jobs" in out.out
