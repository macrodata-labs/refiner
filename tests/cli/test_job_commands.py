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
                    }
                ],
            }
        }

    def cli_get_job_manifest(self, *, job_id: str) -> dict[str, object]:
        return {"jobId": job_id, "manifest": {"version": 1}}

    def cli_get_job_workers(
        self, *, job_id: str, stage_index: int | None
    ) -> dict[str, object]:
        _ = (job_id, stage_index)
        return {
            "items": [
                {
                    "id": "worker-1",
                    "status": "running",
                    "stageId": "job-1:0",
                    "runningShardCount": 1,
                    "completedShardCount": 2,
                    "startedAt": 1_700_000_001_000,
                    "host": "worker-host",
                }
            ]
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


def test_jobs_list_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_list(
        Namespace(status=None, kind=None, limit=20, cursor=None, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert "job-1" in out.out
    assert "cloud pipeline" in out.out


def test_jobs_get_plain_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(jobs, "_client", lambda: _FakeClient())

    rc = jobs.cmd_jobs_get(Namespace(job_id="job-1", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "Job: cloud pipeline (job-1)" in out.out
    assert "Stages" in out.out


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


def test_jobs_metrics_plain_output_accepts_second_timestamps(
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

    rc = jobs.cmd_jobs_metrics(
        Namespace(
            job_id="job-1",
            stage=0,
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
