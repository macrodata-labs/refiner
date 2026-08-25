from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace

from refiner.cli import debug_sessions
from refiner.cli.debug_sessions import (
    find_session,
    new_session_record,
    remove_session,
    save_session,
    session_creation_lock,
)


class _Client:
    def __init__(self, workspace: str) -> None:
        self.base_url = "https://macrodata.test"
        self.workspace = workspace

    def verify_api_key(self):
        return SimpleNamespace(
            workspace=SimpleNamespace(slug=self.workspace),
            key_id="key-1",
            name="test",
        )


def test_session_registry_is_scoped_by_pipeline_endpoint_and_workspace(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    script = tmp_path / "project" / "pipeline.py"
    script.parent.mkdir()
    script.write_text("pass\n")
    client = _Client("workspace-a")
    record = new_session_record(
        script=script,
        project_root=script.parent,
        script_args=["--rows", "10"],
        job_id="job-1",
        client=client,  # type: ignore[arg-type]
    )

    save_session(record)

    assert find_session(script=script, client=client) == record  # type: ignore[arg-type]
    assert find_session(script=script, client=_Client("workspace-b")) is None  # type: ignore[arg-type]
    remove_session(script=script, client=client)  # type: ignore[arg-type]
    assert find_session(script=script, client=client) is None  # type: ignore[arg-type]


def test_session_creation_lock_serializes_same_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    script = tmp_path / "project" / "pipeline.py"
    script.parent.mkdir()
    script.write_text("pass\n")
    client = _Client("workspace-a")
    waiting = Event()
    acquired = Event()

    def acquire_again() -> None:
        waiting.set()
        with session_creation_lock(script=script, client=client):  # type: ignore[arg-type]
            acquired.set()

    with session_creation_lock(script=script, client=client):  # type: ignore[arg-type]
        contender = Thread(target=acquire_again)
        contender.start()
        assert waiting.wait(timeout=1)
        assert not acquired.wait(timeout=0.05)

    contender.join(timeout=1)
    assert not contender.is_alive()
    assert acquired.is_set()


def test_exclusive_file_lock_uses_windows_backend(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fd: int, mode: int, size: int) -> None:
            assert size == 1
            calls.append(mode)

    monkeypatch.setattr(debug_sessions, "_fcntl", None)
    monkeypatch.setattr(debug_sessions, "_msvcrt", _Msvcrt)

    with debug_sessions._exclusive_file_lock(tmp_path / "session.lock"):
        assert calls == [_Msvcrt.LK_NBLCK]

    assert calls == [_Msvcrt.LK_NBLCK, _Msvcrt.LK_UNLCK]
