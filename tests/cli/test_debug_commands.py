from __future__ import annotations

from argparse import Namespace

from refiner.cli import debug


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

    def cloud_debug_stop(self, **kwargs):
        self.calls.append(("stop", kwargs))
        return {"status": "canceled"}

    def cloud_debug_sync(self, **kwargs):
        self.calls.append(("sync", kwargs))
        return {"files": 1}

    def cloud_debug_doctor(self, **kwargs):
        self.calls.append(("doctor", kwargs))
        return {"status": "ready", "python": {"version": "3.10"}}


def test_debug_commands_forward_to_cloud_api(monkeypatch, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(debug, "MacrodataClient", lambda: client)

    assert debug.cmd_debug_status(Namespace(job_id="job-1", json=False)) == 0
    assert (
        debug.cmd_debug_exec(
            Namespace(
                job_id="job-1",
                exec_command=["--", "python", "-V"],
                workdir=None,
                timeout=20,
            )
        )
        == 0
    )
    assert debug.cmd_debug_run(Namespace(job_id="job-1", max_shards=1, timeout=30)) == 2
    assert debug.cmd_debug_stop(Namespace(job_id="job-1", json=False)) == 0

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
            {"job_id": "job-1", "max_shards": 1, "timeout_secs": 30},
        ),
        ("stop", {"job_id": "job-1"}),
    ]
    captured = capsys.readouterr()
    assert "ready\tjob-1" in captured.out
    assert "Python 3.10" in captured.out
    assert "Debug session closed: job-1" in captured.out
    assert "failed" in captured.err


def test_debug_sync_and_doctor(monkeypatch, tmp_path, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(debug, "MacrodataClient", lambda: client)
    (tmp_path / "pipeline.py").write_text("PIPELINE = 1\n")

    assert (
        debug.cmd_debug_sync(Namespace(job_id="job-1", path=str(tmp_path), json=False))
        == 0
    )
    assert debug.cmd_debug_doctor(Namespace(job_id="job-1", json=False)) == 0

    sync_call = client.calls[0]
    assert sync_call[0] == "sync"
    assert sync_call[1]["job_id"] == "job-1"
    assert isinstance(sync_call[1]["archive"], bytes)
    sha256 = sync_call[1]["sha256"]
    assert isinstance(sha256, str)
    assert len(sha256) == 64
    assert client.calls[1] == ("doctor", {"job_id": "job-1"})
    assert "Synced 1 files" in capsys.readouterr().out
