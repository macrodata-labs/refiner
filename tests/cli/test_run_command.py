from __future__ import annotations

from argparse import Namespace

import pytest

from refiner.cli.run import command as run
from refiner.cli.run.cloud import CloudAttachDetached
from refiner.cli.run.local import LocalLaunchInterrupted, LocalLaunchResumeError
from refiner.cli.ui.console import resolve_log_mode


def test_cmd_run_sets_env_overrides_and_forwards_args(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_path(path: str, *, run_name: str):
        captured["path"] = path
        captured["run_name"] = run_name
        captured["argv"] = list(run.sys.argv)
        captured["logs"] = run.os.environ.get("REFINER_LOGS")
        captured["attach"] = run.os.environ.get("REFINER_ATTACH")

    monkeypatch.setattr(run.runpy, "run_path", _fake_run_path)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=["--", "--rows", "10"],
            logs="one",
            attach=True,
            detach=False,
        )
    )

    assert rc == 0
    assert captured["path"] == str(script)
    assert captured["run_name"] == "__main__"
    assert captured["argv"] == [str(script), "--rows", "10"]
    assert captured["logs"] == "one"
    assert captured["attach"] == "attach"
    assert run.os.environ.get("REFINER_ATTACH") is None


def test_cmd_run_sets_auto_attach_mode_by_default(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_path(path: str, *, run_name: str):
        del path, run_name
        captured["attach"] = run.os.environ.get("REFINER_ATTACH")

    monkeypatch.setattr(run.runpy, "run_path", _fake_run_path)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    assert rc == 0
    assert captured["attach"] == "auto"
    assert run.os.environ.get("REFINER_ATTACH") is None


def test_cmd_run_restores_attach_env(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setenv("REFINER_ATTACH", "detach")

    def _fake_run_path(path: str, *, run_name: str):
        captured["attach"] = run.os.environ.get("REFINER_ATTACH")

    monkeypatch.setattr(run.runpy, "run_path", _fake_run_path)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=True,
            detach=False,
        )
    )

    assert rc == 0
    assert captured["attach"] == "attach"
    assert run.os.environ.get("REFINER_ATTACH") == "detach"


def test_cmd_run_missing_script_returns_error(capsys, tmp_path) -> None:
    rc = run.cmd_run(
        Namespace(
            script=str(tmp_path / "missing.py"),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )
    out = capsys.readouterr()
    assert rc == 1
    assert "Script not found:" in out.err


def test_resolve_log_mode_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("REFINER_LOGS", "errors")
    assert resolve_log_mode(None) == "errors"


def test_cmd_run_prints_runtime_error_without_traceback(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise LocalLaunchResumeError("boom")

    monkeypatch.setattr(run.runpy, "run_path", _raise)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    out = capsys.readouterr()
    assert rc == 1
    assert out.err.strip() == "boom"


def test_cmd_run_suppresses_resume_error_print_on_tty(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise LocalLaunchResumeError("boom")

    monkeypatch.setattr(run.runpy, "run_path", _raise)
    monkeypatch.setattr(run.sys.stdout, "isatty", lambda: True)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    out = capsys.readouterr()
    assert rc == 1
    assert out.err == ""


def test_cmd_run_does_not_swallow_plain_runtime_error(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise RuntimeError("boom")

    monkeypatch.setattr(run.runpy, "run_path", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        run.cmd_run(
            Namespace(
                script=str(script),
                script_args=[],
                logs=None,
                attach=False,
                detach=False,
            )
        )


def test_cmd_run_returns_130_for_launcher_interrupt(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise LocalLaunchInterrupted(
            "Local job interrupted. To resume completed shards, rerun with rundir='x'."
        )

    monkeypatch.setattr(run.runpy, "run_path", _raise)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    out = capsys.readouterr()
    assert rc == 130
    assert "Local job interrupted" in out.err


def test_cmd_run_suppresses_generic_interrupt_message_for_cloud_detach(
    monkeypatch, tmp_path, capsys
) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise CloudAttachDetached()

    monkeypatch.setattr(run.runpy, "run_path", _raise)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    out = capsys.readouterr()
    assert rc == 130
    assert out.err == ""


def test_cmd_run_returns_141_for_broken_pipe(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _raise(path: str, *, run_name: str):
        del path, run_name
        raise BrokenPipeError()

    monkeypatch.setattr(run.runpy, "run_path", _raise)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    assert rc == 141


def test_cmd_run_prepends_script_directory_to_sys_path(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_path(path: str, *, run_name: str):
        captured["path"] = path
        captured["run_name"] = run_name
        captured["sys_path_0"] = run.sys.path[0]

    monkeypatch.setattr(run.runpy, "run_path", _fake_run_path)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    assert rc == 0
    assert captured["path"] == str(script)
    assert captured["run_name"] == "__main__"
    assert captured["sys_path_0"] == str(tmp_path.resolve())


def test_cmd_run_drops_cwd_entry_from_sys_path(monkeypatch, tmp_path) -> None:
    script = tmp_path / "demo.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(run.sys, "path", ["", "/existing"])

    def _fake_run_path(path: str, *, run_name: str):
        captured["path"] = path
        captured["run_name"] = run_name
        captured["sys_path"] = list(run.sys.path)

    monkeypatch.setattr(run.runpy, "run_path", _fake_run_path)

    rc = run.cmd_run(
        Namespace(
            script=str(script),
            script_args=[],
            logs=None,
            attach=False,
            detach=False,
        )
    )

    assert rc == 0
    assert captured["path"] == str(script)
    assert captured["run_name"] == "__main__"
    assert captured["sys_path"] == [str(tmp_path.resolve()), "/existing"]
