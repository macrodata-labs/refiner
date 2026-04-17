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
    args = parser.parse_args(["jobs", "list", "--kind", "cloud"])
    assert args.command == "jobs"
    assert args.jobs_command == "list"
    assert args.kind == "cloud"


def test_parser_has_jobs_cancel_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["jobs", "cancel", "job-1"])
    assert args.command == "jobs"
    assert args.jobs_command == "cancel"
    assert args.job_id == "job-1"


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
