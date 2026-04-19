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
                        "runtimeConfig": {
                            "requestedNumWorkers": 4,
                            "cpuCores": 8,
                            "memoryMb": 16384,
                            "gpuCount": 1,
                            "gpuType": "a10g",
                        },
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
                    "endedAt": None,
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
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: False)

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "job-1" in out.out
    assert "cloud pipeline" in out.out
    assert "alex@example.com" in out.out
    assert "\x1b[" not in out.out


def test_jobs_list_plain_output_shows_next_cursor_command(monkeypatch, capsys) -> None:
    class _CursorClient(_FakeClient):
        def cli_list_jobs(self, **_: object) -> dict[str, object]:
            payload = super().cli_list_jobs()
            payload["nextCursor"] = "123"
            return payload

    _patch_job_client(monkeypatch, lambda: _CursorClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: False)

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Next cursor: macrodata jobs list --limit 20 --cursor 123" in out.out


def test_jobs_list_colors_status_kind_for_interactive_terminals(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: True)

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b[1;38;5;220m● running\x1b[0m" in out.out
    assert "\x1b[1;38;5;117mcloud\x1b[0m" in out.out
    assert "\x1b[38;5;255m2023-11-14 22:13:20 UTC\x1b[0m" in out.out
    assert "\x1b[1;38;5;255mcloud pipeline\x1b[0m" in out.out


def test_jobs_get_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: False)

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Job: cloud pipeline  ID: job-1  URL:" in out.out
    assert "Available: manifest, logs, metrics" in out.out
    assert "Status: running  Kind: cloud  Cost: $1.25" in out.out
    assert "Workers:" not in out.out
    assert "Stages" in out.out
    assert "Steps" in out.out
    assert "run=2 done=1 tot=4" in out.out
    assert "Req" in out.out
    assert "CPU" in out.out
    assert "Memory" in out.out
    assert "GPU" in out.out
    assert "  4  " in out.out
    assert "  8  " in out.out
    assert "16384" in out.out
    assert "1 a10g" in out.out
    assert "Progress:" not in out.out
    assert "columns=18" in out.out
    assert "__meta" not in out.out
    assert "\x1b[" not in out.out


def test_jobs_get_plain_output_shows_rundir_for_local_jobs(monkeypatch, capsys) -> None:
    class _LocalClient(_FakeClient):
        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            job = cast(dict[str, object], payload["job"])
            job["executorKind"] = "local"
            job["rundir"] = "/tmp/refiner/runs/job-1"
            return payload

    _patch_job_client(monkeypatch, lambda: _LocalClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: False)

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Rundir: /tmp/refiner/runs/job-1" in out.out


def test_jobs_get_colors_status_for_interactive_terminals(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: True)
    monkeypatch.setattr("refiner.cli.jobs.get.stdout_is_interactive", lambda: True)

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b[1;38;5;220m● running\x1b[0m" in out.out
    assert "\x1b[1;38;5;117mcloud\x1b[0m" in out.out
    assert "\x1b[1;38;5;255mcloud pipeline\x1b[0m" in out.out
    assert "\x1b[1;38;5;255mjob-1\x1b[0m" in out.out
    assert "\x1b[38;5;255m2023-11-14 22:13:20 UTC\x1b[0m" in out.out
    assert "\x1b[1;38;5;255m$1.25\x1b[0m" in out.out


def test_jobs_get_colors_error_for_interactive_terminals(monkeypatch, capsys) -> None:
    class _ErrorClient(_FakeClient):
        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["error"] = (
                "Local launcher interrupted"
            )
            return payload

    _patch_job_client(monkeypatch, lambda: _ErrorClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: True)
    monkeypatch.setattr("refiner.cli.jobs.get.stdout_is_interactive", lambda: True)

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b[1;38;5;203mLocal launcher interrupted\x1b[0m" in out.out


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
    assert captured["anchor"] == "earliest"
    assert captured["cursor"] == "cursor-1"
    assert captured["limit"] == jobs_logs._DEFAULT_LOG_PAGE_LIMIT


def test_jobs_logs_defaults_to_latest_wide_page(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class _FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(1_700_000_000, tz=tz)

    class _LatestClient(_FakeClient):
        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return super().cli_get_job_logs(**kwargs)

    _patch_job_client(monkeypatch, lambda: _LatestClient())
    monkeypatch.setattr(jobs_logs, "datetime", _FixedDateTime)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=None,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=None,
            end_ms=None,
            cursor=None,
            limit=None,
            follow=False,
            json=False,
        )
    )
    _ = capsys.readouterr()

    assert rc == 0
    assert captured["anchor"] == "latest"
    assert captured["start_ms"] == 0
    assert captured["end_ms"] == 1_700_000_000_000
    assert captured["cursor"] is None
    assert captured["limit"] == jobs_logs._DEFAULT_LOG_PAGE_LIMIT


def test_jobs_logs_follow_requests_latest_anchor(monkeypatch, capsys) -> None:
    captured_calls: list[dict[str, object]] = []

    class _FollowClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            captured_calls.append(dict(kwargs))
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
                            "line": "hello",
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

    _patch_job_client(monkeypatch, lambda: _FollowClient())
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
    assert len(captured_calls) >= 2
    assert captured_calls[0]["anchor"] == "latest"
    assert captured_calls[0]["start_ms"] is None
    assert captured_calls[0]["end_ms"] is None
    assert captured_calls[1]["anchor"] == "earliest"
    assert isinstance(captured_calls[1]["start_ms"], int)
    assert isinstance(captured_calls[1]["end_ms"], int)


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


def test_jobs_logs_follow_degrades_to_one_shot_when_job_already_terminal(
    monkeypatch, capsys
) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            return {
                "entries": [
                    {
                        "ts": 1_700_000_003_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "one-shot line",
                        "messageHash": "final",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    client = _TerminalClient()
    _patch_job_client(monkeypatch, lambda: client)

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
    assert "one-shot line" in out.out
    assert (
        "job is already terminal; showing one-shot logs instead of follow mode."
        in out.err
    )
    assert client.log_calls == 1


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
            if self.log_calls >= 1:
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
            if self.log_calls >= 1:
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


def test_jobs_logs_follow_resets_terminal_retry_counter_after_success(
    monkeypatch, capsys
) -> None:
    class _TerminalClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0
            self.final_page_index = 0

        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            self.log_calls += 1
            if self.log_calls == 1:
                return {
                    "entries": [],
                    "hasOlder": False,
                    "nextCursor": None,
                }
            self.final_page_index += 1
            if self.final_page_index == 1:
                raise MacrodataApiError(status=503, message="temporary-1")
            if self.final_page_index == 2:
                return {
                    "entries": [
                        {
                            "ts": 1_700_000_003_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "final page 1",
                            "messageHash": "final-1",
                        }
                    ],
                    "hasOlder": True,
                    "nextCursor": "cursor-final-1",
                }
            if self.final_page_index == 3:
                raise MacrodataApiError(status=503, message="temporary-2")
            return {
                "entries": [
                    {
                        "ts": 1_700_000_004_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "final page 2",
                        "messageHash": "final-2",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 1:
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
    assert "final page 1" in out.out
    assert "final page 2" in out.out


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
    assert client.log_calls == 4


def test_jobs_logs_follow_terminal_drain_dedupes_bootstrap_entries(
    monkeypatch, capsys
) -> None:
    class _TerminalOrderingClient(_FakeClient):
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
                            "ts": 1_700_000_006_000,
                            "severity": "info",
                            "workerId": "worker-1",
                            "sourceType": "worker",
                            "sourceName": "runner",
                            "line": "latest",
                            "messageHash": "6",
                        }
                    ],
                    "hasOlder": True,
                    "nextCursor": "cursor-final",
                }
            return {
                "entries": [
                    {
                        "ts": 1_700_000_006_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "latest",
                        "messageHash": "6",
                    },
                    {
                        "ts": 1_700_000_007_000,
                        "severity": "info",
                        "workerId": "worker-1",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "newer",
                        "messageHash": "7",
                    },
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job(job_id=job_id)
            if self.log_calls >= 1:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    _patch_job_client(monkeypatch, lambda: _TerminalOrderingClient())
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
    assert out.out.count("latest") == 1
    assert "newer" in out.out


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
            if self.job_calls >= 3:
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
    assert client.job_calls == 3


def test_jobs_logs_follow_skips_backlog_to_stay_live(monkeypatch, capsys) -> None:
    class _SkippingClient(_FakeClient):
        def __init__(self) -> None:
            self.log_calls = 0

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            self.log_calls += 1
            cursor = kwargs.get("cursor")
            if self.log_calls <= jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS + 1:
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
            if self.log_calls > jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS + 2:
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
            if self.log_calls > jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS + 1:
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
            if self.job_calls >= jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS + 1:
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
    assert sleep_calls["count"] == jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS - 1
    assert client.job_calls == jobs_logs._FOLLOW_LOG_MAX_DRAIN_POLLS + 1


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
            if self.job_calls >= 2:
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
            if self.job_calls >= 2:
                cast(dict[str, object], payload["job"])["status"] = "completed"
            return payload

    client = _FlakyStatusClient()
    _patch_job_client(monkeypatch, lambda: client)
    now_values = iter([10, 15, 16, 17])

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
    assert client.log_calls == [(None, None), (15000, 16000), (16000, 17000)]


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
            if self.log_calls >= 1:
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
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=None,
            metric=[],
            workers=False,
            worker=[],
            asc=False,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Step 2: enrich_records (map)" in out.out
    assert "rows_processed" in out.out
    assert "Detail: inventory" in out.out
    assert (
        "rerun with --step <index> --metric <label> to fetch metric values" in out.out
    )


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
    assert "  0  " in out.out
    assert "Shards Running" in out.out
    assert "Shards Done" in out.out
    assert "Ended" in out.out
    assert (
        "Next cursor: macrodata jobs workers job-1 --stage 0 --limit 20 --cursor 20"
        in out.out
    )


def test_jobs_logs_plain_output_collapses_redundant_source_name(
    monkeypatch, capsys
) -> None:
    class _LogsClient(_FakeClient):
        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            payload = super().cli_get_job_logs()
            entries = cast(list[dict[str, object]], payload["entries"])
            entries[0]["sourceName"] = "worker"
            return payload

    _patch_job_client(monkeypatch, lambda: _LogsClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: False)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=0,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_002_000,
            cursor=None,
            limit=1,
            follow=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "worker=worker-1 " in out.out
    assert " | WARNING  | retrying shard" in out.out
    assert "worker:worker" not in out.out


def test_jobs_logs_interactive_output_tolerates_unknown_severity(
    monkeypatch, capsys
) -> None:
    class _UnknownSeverityClient(_FakeClient):
        def cli_get_job_logs(self, **_: object) -> dict[str, object]:
            payload = super().cli_get_job_logs()
            entries = cast(list[dict[str, object]], payload["entries"])
            entries[0]["severity"] = "notice"
            entries[0]["workerId"] = "worker-abcdef123456"
            entries[0]["sourceName"] = "worker"
            return payload

    _patch_job_client(monkeypatch, lambda: _UnknownSeverityClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: True)
    monkeypatch.setattr("refiner.cli.jobs.logs.stdout_is_interactive", lambda: True)

    rc = jobs.cmd_jobs_logs(
        Namespace(
            job_id="job-1",
            stage=0,
            worker=None,
            source_type=None,
            source_name=None,
            severity=None,
            search=None,
            start_ms=1_700_000_000_000,
            end_ms=1_700_000_002_000,
            cursor=None,
            limit=1,
            follow=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "worker=worker-a...123456" in out.out
    assert "NOTICE" in out.out
    assert "retrying shard" in out.out
    assert "\x1b[" in out.out


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
    assert (
        "Next cursor: macrodata jobs workers job-1 --stage 0 --limit 20 --cursor '[31mboom[0m'"
        in out.out
    )


def test_jobs_manifest_plain_output(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            deps=False,
            code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Runtime" in out.out
    assert "Dependencies: 1 dependency (rerun with --deps)" in out.out
    assert "Code" in out.out
    assert "Path: pipeline.py" in out.out
    assert "SHA256: abc123" in out.out
    assert "Source: (rerun with --code)" in out.out


def test_jobs_manifest_keeps_runtime_when_extra_sections_requested(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            deps=True,
            code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Runtime" in out.out
    assert "Dependencies: 1 dependency" in out.out
    assert "pandas==2.0.0" in out.out
    assert "Code" in out.out
    assert "Path: pipeline.py" in out.out
    assert "SHA256: abc123" in out.out
    assert "Source: (rerun with --code)" in out.out


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
            deps=False,
            code=True,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Dependencies: 1 dependency (rerun with --deps)" in out.out
    assert "Code" in out.out
    assert "Path: pipeline.py" in out.out
    assert "SHA256: abc123" in out.out
    assert "Source:" in out.out
    assert "\x1b" not in out.out
    assert "\x9b" not in out.out
    assert "print('ok')[31m" in out.out
    assert "next_line()" in out.out


def test_jobs_manifest_json_respects_deps_and_code_flags(monkeypatch, capsys) -> None:
    class _ManifestClient(_FakeClient):
        def cli_get_job_manifest(self, *, job_id: str) -> dict[str, object]:
            payload = super().cli_get_job_manifest(job_id=job_id)
            manifest = cast(dict[str, Any], payload["manifest"])
            manifest["dependencies"] = [
                {"name": "pandas", "version": "2.0.0"},
                {"name": "pyarrow", "version": "17.0.0"},
            ]
            manifest["script"] = {
                "path": "pipeline.py",
                "sha256": "abc123",
                "text": "print('ok')\x1b[31m\nnext_line()",
            }
            return payload

    _patch_job_client(monkeypatch, lambda: _ManifestClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            deps=False,
            code=False,
            json=True,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert '"dependencyCount": 2' in out.out
    assert '"dependencies"' not in out.out
    assert '"text"' not in out.out

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            deps=True,
            code=True,
            json=True,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert '"dependencyCount": 2' in out.out
    assert '"dependencies": "pandas==2.0.0\\npyarrow==17.0.0"' in out.out
    assert '"text": "print(\'ok\')[31m\\nnext_line()"' in out.out


def test_jobs_manifest_interactive_output_styles_sections_and_labels(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())
    monkeypatch.setattr("refiner.cli.jobs.common.stdout_is_interactive", lambda: True)

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            deps=False,
            code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "\x1b[1;38;5;117mRuntime\x1b[0m" in out.out
    assert "\x1b[1;38;5;117mDependencies\x1b[0m" in out.out
    assert "\x1b[1;38;5;117mCode\x1b[0m" in out.out
    assert "\x1b[38;5;245mPython\x1b[0m:" in out.out
    assert "\x1b[38;5;245mPath\x1b[0m:" in out.out


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
            workers=False,
            worker=[],
            asc=False,
            desc=False,
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
            workers=False,
            worker=[],
            asc=False,
            desc=False,
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
            workers=False,
            worker=[],
            asc=False,
            desc=False,
            json=False,
        )
    )

    assert rc == 0
    assert observed["metric_labels"] == ["rows_processed", "queue_depth"]


def test_jobs_metrics_values_plain_output_uses_explicit_columns(
    monkeypatch, capsys
) -> None:
    class _ValuesClient(_FakeClient):
        def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
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

    _patch_job_client(monkeypatch, lambda: _ValuesClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed"],
            workers=False,
            worker=[],
            asc=False,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "rows_processed" in out.out
    assert "(counter)" in out.out
    assert "Total:" in out.out
    assert "Rate (lifetime):" in out.out
    assert "Per Worker (lifetime):" in out.out


def test_jobs_metrics_values_plain_output_renders_worker_rankings(
    monkeypatch, capsys
) -> None:
    class _RankingsClient(_FakeClient):
        def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
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
                        "rankings": [
                            {
                                "metricKind": "counter",
                                "label": "rows_processed",
                                "workers": [
                                    {
                                        "workerId": "worker-1",
                                        "total": 80,
                                        "ratePerSec": 2.0,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }

    _patch_job_client(monkeypatch, lambda: _RankingsClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed"],
            workers=True,
            worker=[],
            asc=False,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Workers (Rate ↓)" in out.out
    assert "Rate / sec" in out.out
    assert "worker-1" in out.out


def test_jobs_metrics_values_plain_output_renders_ascending_worker_rankings(
    monkeypatch, capsys
) -> None:
    class _RankingsClient(_FakeClient):
        def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
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
                        "rankings": [
                            {
                                "metricKind": "counter",
                                "label": "rows_processed",
                                "workers": [
                                    {
                                        "workerId": "worker-1",
                                        "total": 80,
                                        "ratePerSec": 2.0,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }

    _patch_job_client(monkeypatch, lambda: _RankingsClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed"],
            workers=True,
            worker=[],
            asc=True,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Workers (Rate ↑)" in out.out


def test_jobs_metrics_passes_worker_flags(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class _CapturingClient(_FakeClient):
        def cli_get_job_step_metrics(self, **kwargs: object) -> dict[str, object]:
            observed.update(kwargs)
            return {
                "jobId": "job-1",
                "stageIndex": 0,
                "detailLevel": "values",
                "steps": [],
            }

    _patch_job_client(monkeypatch, lambda: _CapturingClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed"],
            workers=True,
            worker=["worker-1", "worker-1", "worker-2"],
            asc=True,
            desc=False,
            json=False,
        )
    )

    assert rc == 0
    assert observed["workers"] is True
    assert observed["worker_ids"] == ["worker-1", "worker-2"]
    assert observed["sort"] == "asc"


def test_jobs_metrics_rejects_worker_without_metric(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=[],
            workers=False,
            worker=["worker-1"],
            asc=False,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert "--worker requires --metric." in out.err


def test_jobs_metrics_rejects_sort_without_workers(monkeypatch, capsys) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=2,
            metric=["rows_processed"],
            workers=False,
            worker=[],
            asc=True,
            desc=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert "--asc/--desc require --workers." in out.err


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


def test_jobs_resource_metrics_rejects_out_of_range_bucket_count(
    monkeypatch, capsys
) -> None:
    _patch_job_client(monkeypatch, lambda: _FakeClient())

    rc = jobs.cmd_jobs_resource_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            worker_id=[],
            range="1h",
            start_ms=None,
            end_ms=None,
            bucket_count=5,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "--bucket-count must be between 20 and 240." in out.err


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


def test_print_table_handles_ansi_colored_cells(capsys) -> None:
    jobs_common._print_table(
        [
            ["\x1b[38;5;245mIdx\x1b[0m", "\x1b[38;5;245mStatus\x1b[0m", "Name"],
            ["0", "\x1b[1;38;5;77mcompleted\x1b[0m", "stage_0"],
        ]
    )
    out = capsys.readouterr()
    lines = out.out.splitlines()

    assert len(lines) >= 3
    assert "Idx" in lines[0]
    assert "Status" in lines[0]
    assert "Name" in lines[0]
    assert "0" in lines[2]
    assert "completed" in lines[2]
    assert "stage_0" in lines[2]
