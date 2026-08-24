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
