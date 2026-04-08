from __future__ import annotations

from refiner.platform.client import MacrodataClient
from typing import cast


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


def test_report_worker_started_sends_name_and_host(monkeypatch) -> None:
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
    )

    assert resp.worker_id == "worker-1"
    assert captured["path"] == "/api/jobs/job-1/stages/2/workers/start"
    assert cast(dict[str, object], captured["json_payload"]) == {
        "name": "cloud-rank-0",
        "host": "modal",
    }


def test_report_worker_started_sends_parent_provider_call_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"worker_id": "worker-1"}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.report_worker_started(
        job_id="job-1",
        stage_index=2,
        parent_provider_call_id="ap-parent/fc-parent",
    )

    assert cast(dict[str, object], captured["json_payload"]) == {
        "parent_provider_call_id": "ap-parent/fc-parent",
    }


def test_start_worker_services_posts_services(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "services": [{"name": "vllm", "kind": "llm", "endpoint": "http://vllm"}]
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.start_worker_services(
        job_id="job-1",
        stage_index=2,
        worker_id="worker-1",
        services=[
            {"name": "vllm", "kind": "llm", "config": {"model": "Qwen/Qwen3.5-9B"}}
        ],
    )

    assert response["services"][0]["endpoint"] == "http://vllm"
    assert (
        captured["path"] == "/api/jobs/job-1/stages/2/workers/worker-1/services/start"
    )
    assert captured["timeout_s"] == 600.0
    assert cast(dict[str, object], captured["json_payload"]) == {
        "services": [
            {"name": "vllm", "kind": "llm", "config": {"model": "Qwen/Qwen3.5-9B"}}
        ]
    }


def test_get_worker_services_uses_get(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "services": [
                {
                    "name": "vllm",
                    "kind": "llm",
                    "endpoint": "http://vllm",
                    "status": "ready",
                }
            ]
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.get_worker_services(
        job_id="job-1",
        stage_index=2,
        worker_id="worker-1",
    )

    assert response["services"][0]["status"] == "ready"
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/jobs/job-1/stages/2/workers/worker-1/services"


def test_stop_worker_services_posts_empty_body(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.stop_worker_services(
        job_id="job-1",
        stage_index=2,
        worker_id="worker-1",
    )

    assert response.ok is True
    assert captured["path"] == "/api/jobs/job-1/stages/2/workers/worker-1/services/stop"
    assert cast(dict[str, object], captured["json_payload"]) == {}
