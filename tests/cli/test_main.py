from __future__ import annotations

from refiner.cli.main import build_parser, main


def test_parser_has_auth_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["whoami"])
    assert args.command == "whoami"


def test_main_dispatches(monkeypatch) -> None:
    monkeypatch.setattr("refiner.cli.main.cmd_whoami", lambda args: 7)
    rc = main(["whoami"])
    assert rc == 7


def test_main_no_args_shows_help(capsys) -> None:
    rc = main([])
    out = capsys.readouterr()
    assert rc == 0
    assert "Macrodata CLI" in out.out
