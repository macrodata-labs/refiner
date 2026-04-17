from __future__ import annotations

from argparse import Namespace

from refiner.cli import jobs


class _FakeClient:
    def cli_list_jobs(self, **_: object) -> dict[str, object]:
        return {
            "items": [
                {
                    "id": "job-1",
                    "status": "running",
                    "executorKind": "cloud",
                    "startedByIdentity": "alex@example.com",
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
                "startedByIdentity": "alex@example.com",
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
                }
            ],
            "hasOlder": False,
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

    def cli_cancel_job(self, *, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "requested_operations": 2,
            "canceled_operations": 2,
            "failed_operations": 0,
        }


def test_jobs_list_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "job-1" in out.out
    assert "cloud pipeline" in out.out
    assert "alex@example.com" in out.out


def test_jobs_get_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Job: cloud pipeline (job-1)" in out.out
    assert "Stages" in out.out
    assert "Steps" in out.out


def test_jobs_logs_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

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
            limit=10,
            json=True,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert '"entries"' in out.out
    assert '"retrying shard"' in out.out


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

    monkeypatch.setattr(jobs, "_client", lambda: _SecondsClient())

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


def test_jobs_metrics_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(job_id="job-1", stage_index=0, step=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Step 2: enrich_records (map)" in out.out
    assert "rows_processed" in out.out


def test_jobs_cancel_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_cancel(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Canceled: job-1" in out.out
    assert "Requested: 2" in out.out


def test_jobs_workers_plain_output_shows_next_cursor(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_workers(
        Namespace(job_id="job-1", stage=0, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "worker-1" in out.out
    assert "worker-a" in out.out
    assert "Next cursor: 20" in out.out


def test_jobs_manifest_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_manifest(
        Namespace(
            job_id="job-1",
            show_runtime=False,
            show_deps=False,
            show_code=False,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "Runtime" in out.out
    assert "Dependencies" in out.out
    assert "Code" in out.out


def test_jobs_get_missing_payload_reports_to_stderr(monkeypatch, capsys) -> None:
    class _MissingJobClient(_FakeClient):
        def cli_get_job(self, *, job_id: str) -> dict[str, object]:
            return {"unexpected": job_id}

    monkeypatch.setattr(jobs, "_client", lambda: _MissingJobClient())

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "Job details unavailable." in out.err


def test_jobs_metrics_missing_payload_reports_to_stderr(monkeypatch, capsys) -> None:
    class _MissingMetricsClient(_FakeClient):
        def cli_get_job_step_metrics(self, **_: object) -> dict[str, object]:
            return {"jobId": "job-1"}

    monkeypatch.setattr(jobs, "_client", lambda: _MissingMetricsClient())

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage_index=0,
            step=None,
            json=False,
        )
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "Metrics unavailable." in out.err


def test_jobs_resource_metrics_rejects_too_many_worker_ids(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

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

    monkeypatch.setattr(jobs, "_client", lambda: _CapturingClient())

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
            raise jobs.MacrodataApiError(status=500, message="boom")

    monkeypatch.setattr(jobs, "_client", lambda: _FailingClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, me=False, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 1
    assert out.out == ""
    assert "HTTP 500: boom" in out.err


def test_print_table_handles_ragged_rows(capsys) -> None:
    jobs._print_table([["A", "B", "C"], ["1", "2"], ["3"]])
    out = capsys.readouterr()

    assert "A  B  C" in out.out
    assert "1  2" in out.out
