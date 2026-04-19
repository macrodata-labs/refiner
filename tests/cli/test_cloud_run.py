from __future__ import annotations

import itertools
from typing import cast

import pytest

from refiner.cli.run import cloud as cloud_run
from refiner.platform.client import MacrodataApiError, MacrodataClient


class _FakeConsole:
    def __init__(self, **_: object) -> None:
        self.kwargs = dict(_)
        self.snapshots: list[object] = []
        self.system_messages: list[str] = []
        self.emitted_lines: list[tuple[str, list[str]]] = []
        self.closed = 0

    def emit_lines(self, *, worker_id: str, lines: list[str]) -> None:
        self.emitted_lines.append((worker_id, list(lines)))

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


def test_active_stage_prefers_failed_started_stage_over_later_queued_stage() -> None:
    stage_index, total_stages = cloud_run._active_stage(
        {
            "status": "failed",
            "stages": [
                {"index": 0, "status": "completed"},
                {"index": 1, "status": "failed"},
                {"index": 2, "status": "queued"},
            ],
        }
    )

    assert stage_index == 1
    assert total_stages == 3


def test_build_snapshot_preserves_stage_zero_progress() -> None:
    context = cloud_run.CloudAttachContext(
        job_id="job-1",
        job_name="cloud pipeline",
        tracking_url="https://example.com/jobs/job-1",
        stage_index=0,
    )

    snapshot = cloud_run._build_snapshot(
        context=context,
        job_payload={
            "job": {
                "id": "job-1",
                "name": "cloud pipeline",
                "status": "running",
                "createdAt": 1_700_000_000_000,
                "startedAt": 1_700_000_001_000,
                "runningWorkers": 2,
                "totalWorkers": 4,
                "stages": [
                    {
                        "index": 0,
                        "status": "running",
                        "completedWorkers": 3,
                        "totalWorkers": 5,
                    },
                    {
                        "index": 1,
                        "status": "queued",
                        "completedWorkers": 0,
                        "totalWorkers": 7,
                    },
                ],
            }
        },
    )

    assert snapshot.stage_index == 0
    assert snapshot.worker_completed == 3
    assert snapshot.stage_workers == 5


def _log_entry(
    *, ts: int, worker_id: str, severity: str, line: str
) -> dict[str, object]:
    return {
        "ts": ts,
        "severity": severity,
        "line": line,
        "workerId": worker_id,
        "sourceType": "worker",
        "sourceName": "runner",
        "messageHash": f"{worker_id}-{ts}-{severity}-{line}",
    }


def test_attach_to_cloud_job_follows_active_stage(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0
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
    created_consoles: list[_FakeConsole] = []
    monkeypatch.setattr(
        cloud_run,
        "LocalStageConsole",
        lambda **kwargs: (
            created_consoles.append(_FakeConsole(**kwargs)) or created_consoles[-1]
        ),
    )
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
    assert created_consoles[-1].closed == 1
    assert created_consoles[-1].kwargs["rundir"] is None


def test_attach_to_cloud_job_uses_console_without_tty(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="hello",
                    )
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert fake_console.emitted_lines


def test_attach_to_cloud_job_rejects_invalid_log_mode_env(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="running")

    monkeypatch.setenv("REFINER_LOCAL_LOGS", "bogus")

    with pytest.raises(SystemExit, match="unsupported local log mode"):
        cloud_run.attach_to_cloud_job(
            client=cast(MacrodataClient, _Client()),
            job_id="job-1",
            initial_job_payload=_job_payload(stage_index=0, status="running"),
            stage_index_hint=0,
        )


def test_attach_to_cloud_job_resets_cursor_on_stage_switch(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls: list[dict[str, object]] = []
            self.job_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            self.job_calls += 1
            if self.job_calls == 1:
                return _job_payload(stage_index=1, status="running")
            return _job_payload(stage_index=1, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            self.log_calls.append(dict(kwargs))
            if len(self.log_calls) == 1:
                return {"entries": [], "hasOlder": True, "nextCursor": "cursor-1"}
            return {"entries": [], "hasOlder": False, "nextCursor": None}

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
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
    assert client.log_calls[0]["stage_index"] == 0
    assert client.log_calls[0]["cursor"] is None
    assert client.log_calls[1]["stage_index"] == 1
    assert client.log_calls[1]["cursor"] is None


def test_attach_to_cloud_job_without_logs_does_not_busy_loop(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="running")

    sleep_calls: list[float] = []
    fake_console = _FakeConsole()
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: 0.0)

    def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        raise cloud_run.CloudAttachDetached()

    monkeypatch.setattr(cloud_run.time, "sleep", _fake_sleep)

    payload = _job_payload(stage_index=0, status="running")
    job = cast(dict[str, object], payload["job"])
    job["logsAvailable"] = False

    try:
        cloud_run.attach_to_cloud_job(
            client=cast(MacrodataClient, _Client()),
            job_id="job-1",
            initial_job_payload=payload,
            stage_index_hint=0,
        )
    except cloud_run.CloudAttachDetached:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected CloudAttachDetached")

    assert sleep_calls
    assert sleep_calls[0] >= cloud_run._ATTACH_LOGS_INTERVAL_SECONDS


def test_attach_to_cloud_job_hides_lines_for_none_mode(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="hello",
                    )
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "none")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert fake_console.emitted_lines == []


def test_attach_to_cloud_job_none_mode_skips_log_requests(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            return {"entries": [], "hasOlder": False, "nextCursor": None}

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    client = _Client()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "none")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, client),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert client.log_calls == 0


def test_attach_to_cloud_job_limits_to_one_worker_in_one_mode(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            payload = _job_payload(stage_index=0, status="completed")
            job = cast(dict[str, object], payload["job"])
            job["runningWorkers"] = 2
            job["totalWorkers"] = 2
            stage = cast(list[dict[str, object]], job["stages"])[0]
            stage["runningWorkers"] = 2
            stage["totalWorkers"] = 2
            return payload

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="first",
                    ),
                    _log_entry(
                        ts=1_700_000_002_001,
                        worker_id="worker-2",
                        severity="info",
                        line="second",
                    ),
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "one")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == ["worker-1"]


def test_attach_to_cloud_job_shows_only_errors_in_errors_mode(monkeypatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="info line",
                    ),
                    _log_entry(
                        ts=1_700_000_002_001,
                        worker_id="worker-2",
                        severity="error",
                        line="error line",
                    ),
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "errors")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == ["worker-2"]


def test_attach_to_cloud_job_errors_mode_falls_back_to_log_line_when_severity_missing(
    monkeypatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    {
                        "ts": 1_700_000_002_001,
                        "severity": None,
                        "workerId": "worker-2",
                        "sourceType": "worker",
                        "sourceName": "runner",
                        "line": "2026-04-19 10:00:00 | ERROR    | boom",
                        "messageHash": "missing-severity",
                    }
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "errors")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == ["worker-2"]


def test_attach_to_cloud_job_errors_mode_does_not_spend_worker_cap_on_info_only_workers(
    monkeypatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            payload = _job_payload(stage_index=0, status="completed")
            job = cast(dict[str, object], payload["job"])
            job["runningWorkers"] = 6
            job["totalWorkers"] = 6
            stage = cast(list[dict[str, object]], job["stages"])[0]
            stage["runningWorkers"] = 6
            stage["totalWorkers"] = 6
            return payload

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="info",
                    ),
                    _log_entry(
                        ts=1_700_000_002_001,
                        worker_id="worker-2",
                        severity="info",
                        line="info",
                    ),
                    _log_entry(
                        ts=1_700_000_002_002,
                        worker_id="worker-3",
                        severity="info",
                        line="info",
                    ),
                    _log_entry(
                        ts=1_700_000_002_003,
                        worker_id="worker-4",
                        severity="info",
                        line="info",
                    ),
                    _log_entry(
                        ts=1_700_000_002_004,
                        worker_id="worker-5",
                        severity="error",
                        line="boom",
                    ),
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "errors")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == ["worker-5"]


def test_attach_to_cloud_job_all_mode_suppresses_workers_beyond_cap(
    monkeypatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            payload = _job_payload(stage_index=0, status="completed")
            job = cast(dict[str, object], payload["job"])
            worker_count = cloud_run._ATTACH_MAX_LOGGED_WORKERS + 1
            job["runningWorkers"] = worker_count
            job["totalWorkers"] = worker_count
            stage = cast(list[dict[str, object]], job["stages"])[0]
            stage["runningWorkers"] = worker_count
            stage["totalWorkers"] = worker_count
            return payload

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            entries = [
                _log_entry(
                    ts=1_700_000_002_000 + index,
                    worker_id=f"worker-{index}",
                    severity="info",
                    line=f"line-{index}",
                )
                for index in range(1, cloud_run._ATTACH_MAX_LOGGED_WORKERS + 2)
            ]
            return {"entries": entries, "hasOlder": False, "nextCursor": None}

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setenv("REFINER_LOCAL_LOGS", "all")
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == [
        f"worker-{index}"
        for index in range(1, cloud_run._ATTACH_MAX_LOGGED_WORKERS + 1)
    ]


def test_attach_to_cloud_job_keeps_polling_logs_when_status_refresh_retries(
    monkeypatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.base_url = "https://example.com"
            self.job_calls = 0
            self.log_calls = 0

        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            del job_id
            self.job_calls += 1
            if self.job_calls == 1:
                raise MacrodataApiError(status=503, message="temporary")
            return _job_payload(stage_index=0, status="completed")

        def cli_get_job_logs(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.log_calls += 1
            if self.log_calls > 1:
                return {"entries": [], "hasOlder": False, "nextCursor": None}
            return {
                "entries": [
                    _log_entry(
                        ts=1_700_000_002_000,
                        worker_id="worker-1",
                        severity="info",
                        line="hello",
                    )
                ],
                "hasOlder": False,
                "nextCursor": None,
            }

    monotonic_values = itertools.count(0.0, 1.0)
    fake_console = _FakeConsole()
    monkeypatch.setattr(cloud_run, "LocalStageConsole", lambda **_: fake_console)
    monkeypatch.setattr(cloud_run.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_run.time, "sleep", lambda _: None)

    rc = cloud_run.attach_to_cloud_job(
        client=cast(MacrodataClient, _Client()),
        job_id="job-1",
        initial_job_payload=_job_payload(stage_index=0, status="running"),
        stage_index_hint=0,
    )

    assert rc == 0
    assert [worker_id for worker_id, _ in fake_console.emitted_lines] == ["worker-1"]
