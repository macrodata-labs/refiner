from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.platform.client import WorkerConfig


def test_create_job_treats_whitespace_workspace_slug_as_none(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "refiner.platform.client.api.request_json",
        lambda **_: {
            "job": {
                "id": "job-1",
                "stages": [{"index": 0}],
                "workspaceSlug": "   ",
            }
        },
    )

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    context = client.create_job(
        name="Job",
        executor={"type": "refiner-local"},
        plan={"stages": [{"name": "stage_0", "steps": []}]},
        manifest={"version": 1},
    )

    assert context.job_id == "job-1"
    assert context.stage_index == 0
    assert context.workspace_slug is None


def test_report_worker_started_sends_worker_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"worker_id": "worker-1"}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    resp = client.report_worker_started(
        job_id="job-1",
        stage_index=2,
        worker_name="cloud-rank-0",
        host="modal",
        config=WorkerConfig(
            cpu_cores=1,
            memory_mb=2048,
            gpu_count=1,
            gpu_type="h100",
        ),
    )

    assert resp.worker_id == "worker-1"
    assert captured["path"] == "/api/jobs/job-1/stages/2/workers/start"
    assert cast(dict[str, object], captured["json_payload"]) == {
        "name": "cloud-rank-0",
        "host": "modal",
        "config": {
            "cpu_cores": 1,
            "memory_mb": 2048,
            "gpu_count": 1,
            "gpu_type": "h100",
        },
    }
