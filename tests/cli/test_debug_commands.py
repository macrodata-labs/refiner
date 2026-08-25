from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
from typing import cast

import pytest

from refiner.cli import debug
from refiner.platform.client import MacrodataApiError, MacrodataClient


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def cloud_debug_status(self, **kwargs):
        self.calls.append(("status", kwargs))
        return {"status": "ready"}

    def cloud_debug_exec(self, **kwargs):
        self.calls.append(("exec", kwargs))
        return {"exit_code": 0, "stdout": "Python 3.10\n", "stderr": ""}

    def cloud_debug_run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return {"exit_code": 2, "stdout": "", "stderr": "failed\n"}

    def cloud_debug_profile(self, **kwargs):
        self.calls.append(("profile", kwargs))
        return {"profile": "<svg>profile</svg>"}

    def cloud_debug_stop(self, **kwargs):
        self.calls.append(("stop", kwargs))
        return {"status": "canceled"}

    def cloud_debug_doctor(self, **kwargs):
        self.calls.append(("doctor", kwargs))
        return {"status": "ready", "python": {"version": "3.10"}}


def _args(command: str, **kwargs) -> Namespace:
    return Namespace(debug_command=command, pipeline=None, job_id="job-1", **kwargs)


def test_debug_commands_forward_to_cloud_api(monkeypatch, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(debug, "MacrodataClient", lambda: client)

    assert debug._dispatch(_args("status", json=False)) == 0
    assert (
        debug._dispatch(
            _args(
                "exec",
                exec_command=["--", "python", "-V"],
                workdir=None,
                timeout=20,
            )
        )
        == 0
    )
    assert (
        debug._dispatch(
            _args(
                "run",
                max_shards=1,
                timeout=30,
                profile=False,
                profile_output="unused.svg",
            )
        )
        == 2
    )
    assert debug._dispatch(_args("stop", json=False)) == 0

    assert client.calls == [
        ("status", {"job_id": "job-1"}),
        (
            "exec",
            {
                "job_id": "job-1",
                "command": ["python", "-V"],
                "workdir": None,
                "timeout_secs": 20,
            },
        ),
        (
            "run",
            {
                "job_id": "job-1",
                "max_shards": 1,
                "timeout_secs": 30,
                "profile": False,
            },
        ),
        ("stop", {"job_id": "job-1"}),
    ]
    captured = capsys.readouterr()
    assert "ready\tjob-1" in captured.out
    assert "Python 3.10" in captured.out
    assert "Debug session closed: job-1" in captured.out
    assert "failed" in captured.err


def test_debug_run_profiles_and_downloads_flamegraph(
    monkeypatch, tmp_path, capsys
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(debug, "MacrodataClient", lambda: client)
    output = tmp_path / "attempt.svg"

    assert (
        debug._dispatch(
            _args(
                "run",
                max_shards=1,
                timeout=30,
                profile=True,
                profile_output=str(output),
            )
        )
        == 2
    )
    assert output.read_text() == "<svg>profile</svg>"
    assert client.calls[-1] == ("profile", {"job_id": "job-1"})
    assert f"Profile saved to {output}" in capsys.readouterr().out


def test_profile_download_uses_a_unique_temporary_file(monkeypatch, tmp_path) -> None:
    temporary_paths: list[Path] = []
    named_temporary_file = debug.tempfile.NamedTemporaryFile

    def record_temporary_file(*args, **kwargs):
        temporary_file = named_temporary_file(*args, **kwargs)
        temporary_paths.append(Path(temporary_file.name))
        return temporary_file

    monkeypatch.setattr(debug.tempfile, "NamedTemporaryFile", record_temporary_file)
    output = tmp_path / "profile.svg"

    debug._save_profile({"profile": "<svg>first</svg>"}, str(output))
    debug._save_profile({"profile": "<svg>second</svg>"}, str(output))

    assert output.read_text() == "<svg>second</svg>"
    assert len(set(temporary_paths)) == 2
    assert all(not path.exists() for path in temporary_paths)


def test_file_driven_parser_supports_create_sync_and_exec() -> None:
    create = debug._parse_debug_args(["pipeline.py"])
    assert create.debug_command == "create"
    assert create.pipeline == "pipeline.py"

    explicit_create = debug._parse_debug_args(["create", "pipeline.py"])
    assert explicit_create.debug_command == "create"
    assert explicit_create.pipeline == "pipeline.py"

    sync = debug._parse_debug_args(["sync", "pipeline.py"])
    assert sync.debug_command == "sync"
    assert sync.pipeline == "pipeline.py"
    assert sync.script_args_provided is False

    sync_without_args = debug._parse_debug_args(["sync", "pipeline.py", "--"])
    assert sync_without_args.script_args == []
    assert sync_without_args.script_args_provided is True

    execute = debug._parse_debug_args(
        ["exec", "pipeline.py", "--timeout", "10", "--", "python", "-V"]
    )
    assert execute.exec_command == ["python", "-V"]
    assert execute.timeout == 10


def test_wait_until_ready_treats_canceled_as_terminal(monkeypatch) -> None:
    client = cast(MacrodataClient, _FakeClient())
    monkeypatch.setattr(
        client,
        "cloud_debug_status",
        lambda **_kwargs: {"status": "canceled"},
    )

    with pytest.raises(RuntimeError, match="debug worker did not start: canceled"):
        debug._wait_until_ready(client=client, job_id="job-1", timeout_secs=1200)


def test_debug_create_rejects_invalid_timeout_before_allocating(monkeypatch) -> None:
    monkeypatch.setattr(
        debug,
        "MacrodataClient",
        lambda: pytest.fail("client must not be created for an invalid timeout"),
    )

    with pytest.raises(SystemExit, match="--startup-timeout must be greater than zero"):
        debug._cmd_create(
            Namespace(
                pipeline="pipeline.py",
                script_args=[],
                startup_timeout=0,
            )
        )


def test_debug_create_validates_sync_bundle_before_allocating(
    monkeypatch, tmp_path
) -> None:
    class Launcher:
        def launch_debug(self):
            pytest.fail("worker must not launch when sync validation fails")

    monkeypatch.setattr(debug, "MacrodataClient", object)
    monkeypatch.setattr(debug, "session_creation_lock", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(debug, "find_session", lambda **_kwargs: None)
    monkeypatch.setattr(debug, "_capture_launcher", lambda *_args: Launcher())
    monkeypatch.setattr(debug, "find_project_root", lambda _script: tmp_path)
    monkeypatch.setattr(
        debug,
        "_build_sync_bundle",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("source archive too large")),
    )

    with pytest.raises(ValueError, match="source archive too large"):
        debug._cmd_create(
            Namespace(
                pipeline=str(tmp_path / "pipeline.py"),
                script_args=[],
                startup_timeout=1200,
            )
        )


def test_debug_create_discards_missing_remembered_session(
    monkeypatch, tmp_path
) -> None:
    client = cast(MacrodataClient, _FakeClient())
    record = debug.DebugSessionRecord(
        script_path=str(tmp_path / "pipeline.py"),
        project_root=str(tmp_path),
        script_args=[],
        job_id="missing-job",
        base_url="https://example.test",
        workspace="workspace",
    )
    removed: list[Path] = []
    monkeypatch.setattr(debug, "find_session", lambda **_kwargs: record)
    monkeypatch.setattr(
        client,
        "cloud_debug_status",
        lambda **_kwargs: (_ for _ in ()).throw(
            MacrodataApiError(status=404, message="missing")
        ),
    )
    monkeypatch.setattr(
        debug,
        "remove_session",
        lambda *, script, client: removed.append(Path(script)),
    )

    debug._clear_existing_session(script=tmp_path / "pipeline.py", client=client)

    assert removed == [tmp_path / "pipeline.py"]


def test_debug_create_preserves_remembered_session_on_status_failure(
    monkeypatch, tmp_path
) -> None:
    client = cast(MacrodataClient, _FakeClient())
    record = debug.DebugSessionRecord(
        script_path=str(tmp_path / "pipeline.py"),
        project_root=str(tmp_path),
        script_args=[],
        job_id="unavailable-job",
        base_url="https://example.test",
        workspace="workspace",
    )
    monkeypatch.setattr(debug, "find_session", lambda **_kwargs: record)
    monkeypatch.setattr(
        client,
        "cloud_debug_status",
        lambda **_kwargs: (_ for _ in ()).throw(
            MacrodataApiError(status=503, message="unavailable")
        ),
    )
    monkeypatch.setattr(
        debug,
        "remove_session",
        lambda **_kwargs: pytest.fail("transient errors must preserve the session"),
    )

    with pytest.raises(MacrodataApiError) as error:
        debug._clear_existing_session(script=tmp_path / "pipeline.py", client=client)

    assert error.value.status == 503


def test_debug_create_discards_canceled_remembered_session(
    monkeypatch, tmp_path
) -> None:
    client = cast(MacrodataClient, _FakeClient())
    record = debug.DebugSessionRecord(
        script_path=str(tmp_path / "pipeline.py"),
        project_root=str(tmp_path),
        script_args=[],
        job_id="canceled-job",
        base_url="https://example.test",
        workspace="workspace",
    )
    removed: list[Path] = []
    monkeypatch.setattr(debug, "find_session", lambda **_kwargs: record)
    monkeypatch.setattr(
        client,
        "cloud_debug_status",
        lambda **_kwargs: {"status": "canceled"},
    )
    monkeypatch.setattr(
        debug,
        "remove_session",
        lambda *, script, client: removed.append(Path(script)),
    )

    debug._clear_existing_session(script=tmp_path / "pipeline.py", client=client)

    assert removed == [tmp_path / "pipeline.py"]


def test_capture_requires_exactly_one_cloud_launch(monkeypatch) -> None:
    monkeypatch.setattr(debug, "cmd_run", lambda _args: 0)

    try:
        debug._capture_launcher("pipeline.py", [])
    except ValueError as error:
        assert "found none" in str(error)
    else:
        raise AssertionError("capture without launch should fail")


def test_capture_executes_an_unchanged_pipeline_script(tmp_path) -> None:
    script = tmp_path / "pipeline.py"
    script.write_text(
        "import refiner as mdr\n"
        "mdr.from_items([1, 2]).launch_cloud(name='captured pipeline')\n"
    )

    launcher = debug._capture_launcher(str(script), [])

    assert launcher.name == "captured pipeline"


def test_debug_sync_json_redirects_pipeline_stdout(monkeypatch, capsys) -> None:
    monkeypatch.setattr(debug, "MacrodataClient", lambda: object())
    monkeypatch.setattr(
        debug,
        "_job_for_target",
        lambda _args, _client: ("job-1", None),
    )

    def capture_launcher(_pipeline, _script_args):
        print("pipeline diagnostic")
        return object()

    monkeypatch.setattr(debug, "_capture_launcher", capture_launcher)
    monkeypatch.setattr(debug, "find_project_root", lambda _pipeline: Path("."))
    monkeypatch.setattr(
        debug,
        "_sync_launcher",
        lambda **_kwargs: (
            print("manifest diagnostic"),
            {"status": "ready", "shards": 2},
        )[1],
    )

    assert (
        debug._cmd_sync(
            Namespace(
                pipeline="pipeline.py",
                job_id="job-1",
                script_args=[],
                json=True,
            )
        )
        == 0
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ready", "shards": 2}
    assert captured.err == "pipeline diagnostic\nmanifest diagnostic\n"


def test_debug_sync_explicit_separator_clears_remembered_arguments(
    monkeypatch, tmp_path
) -> None:
    client = object()
    record = debug.DebugSessionRecord(
        script_path=str(tmp_path / "pipeline.py"),
        project_root=str(tmp_path),
        script_args=["--rows", "10"],
        job_id="job-1",
        base_url="https://example.test",
        workspace="workspace",
    )
    captured_args: list[list[str]] = []
    monkeypatch.setattr(debug, "MacrodataClient", lambda: client)
    monkeypatch.setattr(
        debug,
        "_job_for_target",
        lambda _args, _client: ("job-1", record),
    )
    monkeypatch.setattr(
        debug,
        "_capture_launcher",
        lambda _pipeline, script_args: captured_args.append(script_args) or object(),
    )
    monkeypatch.setattr(debug, "_sync_launcher", lambda **_kwargs: {})

    args = debug._parse_debug_args(["sync", str(tmp_path / "pipeline.py"), "--"])
    assert debug._cmd_sync(args) == 0

    assert captured_args == [[]]


def test_debug_reports_expected_errors_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        debug,
        "_dispatch",
        lambda _args: (_ for _ in ()).throw(RuntimeError("not ready")),
    )

    assert (
        debug.cmd_debug(
            Namespace(debug_help=False, debug_args=["status", "pipeline.py"])
        )
        == 1
    )
    assert capsys.readouterr().err == "not ready\n"
