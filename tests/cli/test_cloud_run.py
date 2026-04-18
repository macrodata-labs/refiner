from __future__ import annotations

import itertools
from typing import cast

from refiner.cli import cloud_run
from refiner.platform.client import MacrodataClient


class _FakeConsole:
    def __init__(self, **_: object) -> None:
        self.snapshots: list[object] = []
        self.system_messages: list[str] = []
        self.closed = 0

    def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
        del worker_id, lines

    def emit_system(self, message: str) -> None:
        self.system_messages.append(message)

    def apply_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        self.closed += 1


def _job_payload(*, stage_index: int, status: str) -> dict[str, object]:
    return {
        "job": {
            "id": "job-1",
            "name": "cloud pipeline",
            "status": status,
            "executorKind": "cloud",
            "createdAt": 1_700_000_000_000,
            "startedAt": 1_700_000_001_000,
            "runningWorkers": 1,
            "totalWorkers": 1,
            "logsAvailable": True,
            "stages": [
                {
                    "index": 0,
                    "status": "completed" if stage_index > 0 else status,
                    "completedWorkers": 1 if stage_index > 0 else 0,
                    "runningWorkers": 0 if stage_index > 0 else 1,
                    "totalWorkers": 1,
                },
                {
                    "index": 1,
                    "status": status if stage_index > 0 else "queued",
                    "completedWorkers": 0,
                    "runningWorkers": 1 if stage_index > 0 else 0,
                    "totalWorkers": 1,
                },
            ],
        }
    }


def test_attach_to_cloud_job_follows_active_stage(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.logged_stage_indexes: list[int | None] = []

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=1, status="completed")

        def cli_get_job_workers(
            self,
            *,
            job_id: str,
            stage_index: int | None,
            limit: int | None = None,
            cursor: str | None = None,
        ) -> dict[str, object]:
            del job_id, stage_index, limit, cursor
            return {"items": []}

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            self.logged_stage_indexes.append(
                cast(int | None, kwargs.get("stage_index"))
            )
            return {"entries": [], "hasOlder": False, "nextCursor": None}

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setattr(cloud_run, "stdout_is_interactive", lambda: True)
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    client = _Client()
    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, client),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert client.logged_stage_indexes[-1] == 1
    assert fake_console.closed == 1
