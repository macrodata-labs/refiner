from __future__ import annotations

import builtins
from argparse import Namespace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

from refiner.cli.jobs import common as jobs_common
from refiner.cli.jobs.attach import cmd_jobs_attach
from refiner.cli.jobs.control import cmd_jobs_cancel
from refiner.cli.jobs.get import cmd_jobs_get
from refiner.cli.jobs.list import cmd_jobs_list
from refiner.cli.jobs import logs as jobs_logs
from refiner.cli.jobs import get as jobs_get_module
from refiner.cli.jobs import list as jobs_list_module
from refiner.cli.jobs import attach as jobs_attach_module
from refiner.cli.jobs import manifest as jobs_manifest_module
from refiner.cli.jobs import metrics as jobs_metrics_module
from refiner.cli.jobs import workers as jobs_workers_module
from refiner.cli.jobs import control as jobs_control_module
from refiner.cli.jobs.logs import cmd_jobs_logs
from refiner.cli.jobs.manifest import cmd_jobs_manifest
from refiner.cli.jobs.metrics import cmd_jobs_metrics, cmd_jobs_resource_metrics
from refiner.cli.jobs.workers import cmd_jobs_workers
from refiner.platform.client.api import MacrodataApiError

jobs = SimpleNamespace(
    cmd_jobs_attach=cmd_jobs_attach,
    cmd_jobs_cancel=cmd_jobs_cancel,
    cmd_jobs_get=cmd_jobs_get,
    cmd_jobs_list=cmd_jobs_list,
    cmd_jobs_logs=cmd_jobs_logs,
    cmd_jobs_manifest=cmd_jobs_manifest,
    cmd_jobs_metrics=cmd_jobs_metrics,
    cmd_jobs_resource_metrics=cmd_jobs_resource_metrics,
    cmd_jobs_workers=cmd_jobs_workers,
)


class _FakeClient:
    def cli_list_jobs(self, **_: object) -> dict[str, object]:
        return {
            "items": [
                {
                    "id": "job-1",
                    "status": "running",
                    "executorKind": "cloud",
                    "startedByUsername": "alex",
                    "startedByEmail": "alex@example.com",
                    "progress": {"done": 3, "total": 7},
                    "createdAt": 1_700_000_000_000,
                    "name": "cloud pipeline",
                }
            ],
            "nextCursor": None,
        }

    def cli_get_job(self, *, job_id: str) -> dict[str, object]:
        return {
            "job": {
                "id": job_id,
                "name": "cloud pipeline",
                "status": "running",
                "executorKind": "cloud",
                "startedByUsername": "alex",
                "startedByEmail": "alex@example.com",
                "progress": {"done": 3, "total": 7},
                "createdAt": 1_700_000_000_000,
                "startedAt": 1_700_000_001_000,
                "endedAt": None,
                "runningWorkers": 2,
                "totalWorkers": 4,
                "currentCostUsd": "1.25",
                "manifestAvailable": True,
                "logsAvailable": True,
                "metricsAvailable": True,
                "stages": [
                    {
                        "index": 0,
                        "status": "running",
                        "shardDone": 3,
                        "shardTotal": 10,
                        "runningWorkers": 2,
                        "completedWorkers": 1,
                        "totalWorkers": 4,
                        "name": "stage-0",
                        "steps": [
                            {
                                "index": 0,
                                "name": "normalize_rows",
                                "type": "map",
                                "args": {"columns": 18},
                            }
                        ],
                    }
                ],
            }
        }

    def cli_get_job_manifest(self, *, job_id: str) -> dict[str, object]:
        return {
            "jobId": job_id,
            "manifest": {
                "version": 1,
                "environment": {
                    "python_version": "3.11.0",
                    "refiner_version": "0.1.0",
                    "platform": "linux",
                },
                "dependencies": [{"name": "pandas", "version": "2.0.0"}],
                "script": {
                    "path": "pipeline.py",
                    "sha256": "abc123",
                    "text": "print('hello')",
                },
            },
        }

    def cli_get_job_workers(
        self,
        *,
        job_id: str,
        stage_index: int | None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]:
        _ = (job_id, stage_index, limit, cursor)
        return {
            "items": [
                {
                    "id": "worker-1",
                    "name": "worker-a",
                    "status": "running",
                    "stageId": "job-1:0",
                    "runningShardCount": 1,
                    "completedShardCount": 2,
                    "startedAt": 1_700_000_001_000,
                    "host": "worker-host",
                }
            ],
            "page": {"nextCursor": "20", "loaded": 1, "total": 2},
        }

    def cli_get_job_logs(self, **_: object) -> dict[str, object]:
        return {
            "entries": [
                {
                    "ts": 1_700_000_001_000,
                    "severity": "warning",
                    "workerId": "worker-1",
                    "sourceType": "worker",
                    "sourceName": "runner",
                    "line": "retrying shard",
                    "messageHash": "1",
                }
            ],
            "hasOlder": False,
            "nextCursor": None,
        }

    def cli_get_job_metrics(self, **_: object) -> dict[str, object]:
        return {
            "jobId": "job-1",
            "stageIndex": 0,
            "range": "1h",
            "metrics": {
                "resources": [
                    {
                        "t": 1_700_000_001_000,
                        "cpuUsage": 0.5,
                        "cpuQuota": 1.0,
                        "memoryUsage": 256.0,
                        "memoryLimit": 512.0,
                        "networkInMb": 12.0,
                        "networkOutMb": 2.0,
                    }
                ]
            },
        }

    def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
        return {
            "jobId": "job-1",
            "stageIndex": 0,
            "detailLevel": "inventory",
            "steps": [
                {
                    "stepIndex": 2,
                    "name": "enrich_records",
                    "type": "map",
                    "metrics": [
                        {
                            "metricKind": "counter",
                            "label": "rows_processed",
                            "unit": "rows",
                        }
                    ],
                }
            ],
        }

    def cli_cancel_job(self, *, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "requested_operations": 2,
            "canceled_operations": 2,
            "failed_operations": 0,
        }


def _patch_job_client(monkeypatch, factory) -> None:
    monkeypatch.setattr(jobs_list_module, "_client", factory)
    monkeypatch.setattr(jobs_get_module, "_client", factory)
    monkeypatch.setattr(jobs_attach_module, "_client", factory)
    monkeypatch.setattr(jobs_logs, "_client", factory)
    monkeypatch.setattr(jobs_manifest_module, "_client", factory)
    monkeypatch.setattr(jobs_metrics_module, "_client", factory)
    monkeypatch.setattr(jobs_workers_module, "_client", factory)
    monkeypatch.setattr(jobs_control_module, "_client", factory)


def test_jobs_list_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "job-1" in out.out
    assert "cloud pipeline" in out.out
    assert "alex@example.com" in out.out


def test_jobs_get_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Job: cloud pipeline (job-1)" in out.out
    assert "Stages" in out.out
    assert "Steps" in out.out
    assert "2/1/4" in out.out


def test_jobs_attach_calls_cloud_attach(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_attach_to_cloud_job(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        _fake_attach_to_cloud_job,
    )

    rc = jobs.cmd_jobs_attach(Namespace(job_id="job-1"))

    assert rc == 0
    assert captured["job_id"] == "job-1"
    assert captured.get("force_attach", False) is True
    payload = cast(dict[str, object], captured["initial_job_payload"])
    job = cast(dict[str, object], payload["job"])
    assert job["executorKind"] == "cloud"


def test_jobs_attach_rejects_non_cloud_job(monkeypatch, capsys) -> None:
    class _LocalClient(_FakeClient):
        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["executorKind"] = "local"
            return payload

    _patch_job_client(monkeypatch, lambda: _LocalClient())

    rc = jobs.cmd_jobs_attach(Namespace(job_id="job-1"))
    out = capsys.readouterr()

    assert rc == 1
    assert "only supported for cloud jobs" in out.err


def test_jobs_attach_routes_attach_api_errors_through_cli_handler(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: (_ for _ in ()).throw(
            MacrodataApiError(status=503, message="temporary")
        ),
    )

    rc = jobs.cmd_jobs_attach(Namespace(job_id="job-1"))
    out = capsys.readouterr()

    assert rc == 1
    assert "temporary" in out.err


def test_jobs_attach_routes_attach_system_exit_through_cli_handler(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: (_ for _ in ()).throw(SystemExit("unsupported log mode")),
    )

    rc = jobs.cmd_jobs_attach(Namespace(job_id="job-1"))
    out = capsys.readouterr()

    assert rc == 1
    assert "unsupported log mode" in out.err


def test_jobs_logs_json_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=False,
            json=True,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert '"entries"' in out.out
    assert '"retrying shard"' in out.out


def test_jobs_logs_follow_rejects_json(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=True,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert "--follow cannot be combined with --json." in out.err


def test_jobs_logs_follow_rejects_cursor(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor="cursor-1",
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert "--follow cannot be combined with --cursor." in out.err


def test_jobs_logs_passes_cursor_to_client(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class _CursorClient(_FakeClient):
        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return super().cli_get_job_logs(**kwargs)

    _patch_job_client(monkeypatch, lambda: _CursorClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor="cursor-1",
            limit=None,
            follow=False,
            json=False,
        )
    )
    _ = capsys.readouterr()

    assert rc == 0
    assert captured["cursor"] == "cursor-1"
    assert captured["limit"] == jobs_logs._DEFAULT_LOG_PAGE_LIMIT


def test_jobs_logs_follow_streams_until_interrupted(monkeypatch, capsys) -> None:
    class _FollowClient(_FakeClient):
        def __init__(self) -> None:
            self.calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "warning",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "retrying shard",
                            "messageHash": "1",
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_002_000,
                        "severity": "error",
                        "workerId": "worker-2",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "shard failed",
                        "messageHash": "2",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    client = _FollowClient()
    _patch_job_client(monkeypatch, lambda: client)

    sleep_calls = {"count": 0}

    def _fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(jobs_logs.time, "sleep", _fake_sleep)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 130
    assert "retrying shard" in out.out
    assert "shard failed" in out.out
    assert "Stopped following logs." in out.err


def test_jobs_logs_follow_returns_when_job_is_terminal(monkeypatch, capsys) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "done",
                            "messageHash": "1",
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            return {"entries": [], "hasOlder": False, "nextCursor": None}

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 1:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _TerminalClient()
    _patch_job_client(monkeypatch, lambda: client)

    sleep_calls = {"count": 0}

    def _fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(jobs_logs.time, "sleep", _fake_sleep)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "done" in out.out
    assert out.err == ""
    assert sleep_calls["count"] == 0


def test_jobs_logs_follow_fetches_one_final_window_after_terminal_status(
    monkeypatch, capsys
) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            if self.log_calls == 2:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_003_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "final line",
                            "messageHash": "final",
                        }
                    ],
                    "hasOlder": True,
                    "nextCursor": "cursor-final",
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_004_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "final line 2",
                        "messageHash": "final-2",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _TerminalClient()
    _patch_job_client(monkeypatch, lambda: client)
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "final line" in out.out
    assert "final line 2" in out.out
    assert client.log_calls == 3


def test_jobs_logs_follow_retries_terminal_window_fetch_errors(
    monkeypatch, capsys
) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            if self.log_calls == 2:
                raise MacrodataApiError(status=503, message="temporary")
            return {
                "entries": [
                    {
                        "ts": 1_700_000_003_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "final recovered",
                        "messageHash": "final",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _TerminalClient())
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "final recovered" in out.out


def test_jobs_logs_follow_drains_backlog_before_terminal_exit(
    monkeypatch, capsys
) -> None:
    class _BacklogClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "first",
                            "messageHash": "1",
                        }
                    ],
                    "hasOlder": True,
                    "nextCursor": "cursor-1",
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_002_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "second",
                        "messageHash": "2",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 2:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _BacklogClient()
    _patch_job_client(monkeypatch, lambda: client)
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=1,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "first" in out.out
    assert "second" in out.out


def test_jobs_logs_follow_terminal_drain_breaks_repeated_cursor(
    monkeypatch, capsys
) -> None:
    class _StickyCursorClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "first",
                            "messageHash": "1",
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_002_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": f"final-{self.log_calls}",
                        "messageHash": str(self.log_calls),
                    }
                ],
                "hasOlder": True,
                "nextCursor": "sticky-cursor",
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 1:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _StickyCursorClient()
    _patch_job_client(monkeypatch, lambda: client)
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=1,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "first" in out.out
    assert "final-2" in out.out
    assert client.log_calls == 3


def test_jobs_logs_follow_skips_sleep_while_draining_full_batches(
    monkeypatch, capsys
) -> None:
    class _BusyClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0
            self.job_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "first",
                            "messageHash": "1",
                        },
                        {
                            "ts": 1_700_000_002_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "second",
                            "messageHash": "2",
                        },
                    ],
                    "hasOlder": True,
                    "nextCursor": "cursor-1",
                }
            return {"entries": [], "hasOlder": False, "nextCursor": None}

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            self.job_calls += 1
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 2:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _BusyClient()
    _patch_job_client(monkeypatch, lambda: client)

    sleep_calls = {"count": 0}

    def _fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(jobs_logs.time, "sleep", _fake_sleep)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=2,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "first" in out.out
    assert "second" in out.out
    assert sleep_calls["count"] == 1
    assert client.job_calls == 1


def test_jobs_logs_follow_skips_backlog_to_stay_live(monkeypatch, capsys) -> None:
    class _SkippingClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            self.log_calls += 1
            cursor = kwargs.get("cursor")
            if self.log_calls <= jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_000_000 + self.log_calls,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": f"backlog-{self.log_calls}",
                            "messageHash": str(self.log_calls),
                        }
                    ],
                    "hasOlder": True,
                    "nextCursor": f"cursor-{self.log_calls}",
                }
            if cursor is None:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_010_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "live-now",
                            "messageHash": "live",
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            raise AssertionError("expected cursor reset after backlog skip")

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls > jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _SkippingClient())
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=1,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "backlog-1" in out.out
    assert "live-now" in out.out
    assert "skipped older backlog" in out.err
    assert "request logs for fewer workers" in out.err


def test_jobs_logs_follow_sleeps_after_bounded_drain_polls(monkeypatch, capsys) -> None:
    class _HotClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0
            self.job_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls > jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_000_000 + self.log_calls,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": f"line-{self.log_calls}",
                            "messageHash": str(self.log_calls),
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_000_000 + self.log_calls,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": f"line-{self.log_calls}",
                        "messageHash": str(self.log_calls),
                    }
                ],
                "hasOlder": True,
                "nextCursor": f"cursor-{self.log_calls}",
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            self.job_calls += 1
            payload = super().cli_get_job(job_id=job_id)
            if self.job_calls >= 2:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _HotClient()
    _patch_job_client(monkeypatch, lambda: client)

    sleep_calls = {"count": 0}

    def _fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(jobs_logs.time, "sleep", _fake_sleep)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=1,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "line-1" in out.out
    assert f"line-{jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS}" in out.out
    assert sleep_calls["count"] == jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS
    assert client.job_calls == 2


def test_jobs_logs_follow_ignores_transient_status_probe_errors(
    monkeypatch, capsys
) -> None:
    class _FlakyStatusClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0
            self.job_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_001_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "first",
                            "messageHash": "1",
                        }
                    ],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            return {"entries": [], "hasOlder": False, "nextCursor": None}

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            self.job_calls += 1
            if self.job_calls == 1:
                raise MacrodataApiError(status=503, message="temporary")
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _FlakyStatusClient())
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "first" in out.out


def test_jobs_logs_follow_advances_window_after_transient_status_probe_error(
    monkeypatch,
) -> None:
    class _FlakyStatusClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls: list[tuple[int, int]] = []
            self.job_calls = 0

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            self.log_calls.append(
                (cast(int, kwargs["start_ms"]), cast(int, kwargs["end_ms"]))
            )
            return {"entries": [], "hasOlder": False, "nextCursor": None}

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            self.job_calls += 1
            if self.job_calls == 1:
                raise MacrodataApiError(status=503, message="temporary")
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _FlakyStatusClient()
    _patch_job_client(monkeypatch, lambda: client)
    now_values = iter([10, 15, 16])

    class _FakeDateTime:
        @staticmethod
        def now(*, tz=None):
            return datetime.fromtimestamp(next(now_values), tz=tz)

    monkeypatch.setattr(jobs_logs, "datetime", _FakeDateTime)
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )

    assert rc == 0
    assert client.log_calls == [(1, 2), (2, 10000), (10000, 16000)]


def test_jobs_logs_follow_retries_transient_log_fetch_errors(
    monkeypatch, capsys
) -> None:
    class _FlakyLogsClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                raise MacrodataApiError(status=503, message="temporary")
            return {
                "entries": [
                    {
                        "ts": 1_700_000_001_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "recovered",
                        "messageHash": "1",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 2:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _FlakyLogsClient())
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "recovered" in out.out


def test_jobs_logs_follow_flushes_stream_output(monkeypatch) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            return {
                "entries": [
                    {
                        "ts": 1_700_000_001_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "flushed",
                        "messageHash": "1",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _TerminalClient()
    _patch_job_client(monkeypatch, lambda: client)

    recorded_flushes: list[bool] = []
    original_print = builtins.print

    def _recording_print(*args: Any, **kwargs: Any) -> None:
        recorded_flushes.append(bool(kwargs.get("flush")))
        original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", _recording_print)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )

    assert rc == 0
    assert True in recorded_flushes


def test_jobs_logs_non_follow_returns_clean_interrupt_exit(monkeypatch) -> None:
    class _InterruptClient(_FakeClient):
        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            raise KeyboardInterrupt

    _patch_job_client(monkeypatch, lambda: _InterruptClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=False,
            json=False,
        )
    )

    assert rc == 130


def test_jobs_logs_follow_surfaces_permanent_status_probe_errors(monkeypatch) -> None:
    class _PermanentStatusErrorClient(_FakeClient):
        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            return {"entries": [], "hasOlder": False, "nextCursor": None}

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            raise MacrodataApiError(status=404, message="missing")

    _patch_job_client(monkeypatch, lambda: _PermanentStatusErrorClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )

    assert rc == 1


def test_jobs_logs_follow_uses_larger_default_limit(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class _LimitClient(_FakeClient):
        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "entries": [],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _LimitClient())
    monkeypatch.setattr(jobs_logs.time, "sleep", lambda _: None)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1,
            end_ms=2,
            cursor=None,
            limit=None,
            follow=True,
            json=False,
        )
    )
    _ = capsys.readouterr()

    assert rc == 0
    assert captured["limit"] == jobs_logs._DEFAULT_FOLLOW_LOG_PAGE_LIMIT


def test_jobs_resource_metrics_plain_output_accepts_second_timestamps(
    monkeypatch, capsys
) -> None:
    class _SecondsClient(_FakeClient):
        def cli_get_job_metrics(self, **_: object) -> dict[str, object]:
            return {
                "jobId": "job-1",
                "range": "1h",
                "metrics": {
                    "resources": [
                        {
                            "t": 1_700_000_001,
                            "cpuUsage": 0.5,
                            "cpuQuota": 1.0,
                            "memoryUsage": 256.0,
                            "memoryLimit": 512.0,
                            "networkInMb": 12.0,
                            "networkOutMb": 2.0,
                        }
                    ]
                },
            }

    _patch_job_client(monkeypatch, lambda: _SecondsClient())

    rc = jobs.cmd_jobs_resource_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            worker_id=[],
            range="1h",
            start_ms=None,
            end_ms=None,
            bucket_count=None,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Latest sample: 2023-11-14 22:13:21 UTC" in out.out


def test_jobs_list_plain_output_ignores_invalid_timestamps(monkeypatch, capsys) -> None:
    class _InvalidTimestampClient(_FakeClient):
        def cli_list_jobs(self, **_: object) -> dict[str, object]:
            return {
                "items": [
                    {
                        "id": "job-1",
                        "status": "running",
                        "executorKind": "cloud",
                        "startedByUsername": "alex",
                        "startedByEmail": "alex@example.com",
                        "progress": {"done": 1, "total": 2},
                        "createdAt": float("nan"),
                        "name": "daily parquet enrich",
                    }
                ],
                "nextCursor": None,
            }

    _patch_job_client(monkeypatch, lambda: _InvalidTimestampClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, limit=20, cursor=None, me=False, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "job-1" in out.out
    assert "daily parquet enrich" in out.out
    assert "alex@example.com" in out.out


def test_jobs_metrics_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(job_id="job-1", stage_index=0, step=None, metric=[], json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Step 2: enrich_records (map)" in out.out
    assert "rows_processed" in out.out
    assert "Detail: inventory" in out.out


def test_jobs_cancel_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_cancel(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Canceled: job-1" in out.out
    assert "Requested: 2" in out.out


def test_jobs_cancel_plain_output_accepts_camel_case(monkeypatch, capsys) -> None:
    class _CamelCancelClient(_FakeClient):
        def cli_cancel_job(self, *, job_id: str) -> dict[str, object]:
            return {
                "jobId": job_id,
                "requestedOperations": 2,
                "canceledOperations": 2,
                "failedOperations": 0,
            }

    _patch_job_client(monkeypatch, lambda: _CamelCancelClient())

    rc = jobs.cmd_jobs_cancel(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Canceled: job-1" in out.out
    assert "Requested: 2" in out.out


def test_jobs_workers_plain_output_shows_next_cursor(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_workers(
        Namespace(job_id="job-1", stage=0, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "worker-1" in out.out
    assert "worker-a" in out.out
    assert "Next cursor: 20" in out.out


def test_jobs_workers_plain_output_sanitizes_next_cursor(monkeypatch, capsys) -> None:
    class _CursorClient(_FakeClient):
        def cli_get_job_workers(self, **_: object) -> dict[str, object]:
            payload = super().cli_get_job_workers(job_id="job-1", stage_index=0)
            payload["page"] = {
                "nextCursor": "\x1b[31mboom\x1b[0m",
                "loaded": 1,
                "total": 2,
            }
            return payload

    _patch_job_client(monkeypatch, lambda: _CursorClient())

    rc = jobs.cmd_jobs_workers(
        Namespace(job_id="job-1", stage=0, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b" not in out.out
    assert "Next cursor: [31mboom[0m" in out.out


def test_jobs_manifest_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            show_deps=False,
            show_code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Runtime" in out.out
    assert "Dependencies" not in out.out
    assert "Code" not in out.out


def test_jobs_manifest_keeps_runtime_when_extra_sections_requested(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            show_deps=True,
            show_code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Runtime" in out.out
    assert "Dependencies" in out.out


def test_jobs_manifest_sanitizes_script_text(monkeypatch, capsys) -> None:
    class _ManifestClient(_FakeClient):
        def cli_get_job_manifest(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job_manifest(job_id=job_id)
            manifest = payload["manifest"]
            assert isinstance(manifest, dict)
            manifest_data = cast(dict[str, Any], manifest)
            manifest_data["script"] = {
                "path": "pipeline.py",
                "sha256": "abc123",
                "text": "print('ok')\x1b[31m\x9b31m\nnext_line()",
            }
            return payload

    _patch_job_client(monkeypatch, lambda: _ManifestClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            show_deps=False,
            show_code=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b" not in out.out
    assert "\x9b" not in out.out
    assert "print('ok')[31m" in out.out
    assert "next_line()" in out.out


def test_jobs_get_missing_payload_reports_to_stderr(monkeypatch, capsys) -> None:
    class _MissingJobClient(_FakeClient):
        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            return {"unexpected": job_id}

    _patch_job_client(monkeypatch, lambda: _MissingJobClient())

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "Job details unavailable." in out.err


def test_jobs_metrics_missing_payload_reports_to_stderr(monkeypatch, capsys) -> None:
    class _MissingMetricsClient(_FakeClient):
        def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
            return {"jobId": "job-1"}

    _patch_job_client(monkeypatch, lambda: _MissingMetricsClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=None,
            metric=[],
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "Metrics unavailable." in out.err


def test_jobs_metrics_rejects_metric_without_step(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=None,
            metric=["rows_processed"],
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "--metric requires --step." in out.err


def test_jobs_logs_search_requires_explicit_scope(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search="retry",
            start_ms=None,
            end_ms=None,
            limit=10,
            follow=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "--search requires --stage." in out.err


def test_jobs_logs_search_rejects_large_limits(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=0,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search="retry",
            start_ms=1,
            end_ms=2,
            limit=101,
            follow=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "--search supports at most 100 results." in out.err


def test_jobs_logs_search_requires_explicit_window(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=0,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search="retry",
            start_ms=1,
            end_ms=None,
            limit=10,
            follow=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "--search requires explicit --start-ms and --end-ms." in out.err


def test_jobs_metrics_passes_metric_labels(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class _CapturingClient(_FakeClient):
        def cli_get_job_step_metrics(self, **kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return {
                "jobId": "job-1",
                "stageIndex": 0,
                "detailLevel": "values",
                "steps": [
                    {
                        "stepIndex": 2,
                        "name": "enrich_records",
                        "type": "map",
                        "metrics": [
                            {
                                "metricKind": "counter",
                                "label": "rows_processed",
                                "total": 100,
                                "rateSinceStart": 2.5,
                                "perWorker": 50,
                            }
                        ],
                    }
                ],
            }

    _patch_job_client(monkeypatch, lambda: _CapturingClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed", "rows_processed", "queue_depth"],
            json=False,
        )
    )

    assert rc == 0
    assert observed["metric_labels"] == ["rows_processed", "queue_depth"]


def test_jobs_resource_metrics_rejects_too_many_worker_ids(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_resource_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            worker_id=[f"worker-{index}" for index in range(51)],
            range="1h",
            start_ms=None,
            end_ms=None,
            bucket_count=None,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "Too many --worker-id values; maximum is 50." in out.err


def test_jobs_resource_metrics_deduplicates_worker_ids(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class _CapturingClient(_FakeClient):
        def cli_get_job_metrics(self, **kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return super().cli_get_job_metrics(**kwargs)

    _patch_job_client(monkeypatch, lambda: _CapturingClient())

    rc = jobs.cmd_jobs_resource_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            worker_id=["worker-1", "worker-1", "worker-2"],
            range="1h",
            start_ms=None,
            end_ms=None,
            bucket_count=None,
            json=False,
        )
    )

    assert rc == 0
    assert observed["worker_ids"] == ["worker-1", "worker-2"]


def test_jobs_error_reports_to_stderr(monkeypatch, capsys) -> None:
    class _FailingClient(_FakeClient):
        def cli_list_jobs(self, **_: object) -> dict[str, object]:
            raise MacrodataApiError(status=500, message="boom")

    _patch_job_client(monkeypatch, lambda: _FailingClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "HTTP 500: boom" in out.err


def test_print_table_handles_ragged_rows(capsys) -> None:
    jobs_common._print_table([["A", "B", "C"], ["1", "2"], ["3"]])
    out = capsys.readouterr()

    assert "A  B  C" in out.out
    assert "1  2" in out.out
