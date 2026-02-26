from __future__ import annotations

from refiner.platform.client import MacrodataClient
from refiner.platform.cloud.models import (
    CloudPipelinePayload,
    CloudRunCreateRequest,
    CloudRuntimeConfig,
)
from refiner.platform.http import MacrodataApiError


def _request() -> CloudRunCreateRequest:
    return CloudRunCreateRequest(
        name="demo-cloud-job",
        plan={"stages": [{"name": "stage_0", "steps": []}]},
        runtime=CloudRuntimeConfig(num_workers=2, heartbeat_every_rows=4096),
        pipeline_payload=CloudPipelinePayload(
            format="cloudpickle",
            bytes_b64="AQID",
            sha256="abc123",
            size_bytes=3,
        ),
    )


def test_cloud_client_cloud_submit_job_posts_to_cloud_runs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return {"job_id": "job-1", "stage_id": "stage-1", "status": "queued"}

    monkeypatch.setattr("refiner.platform.client.request_json", fake_request_json)

    client = MacrodataClient(api_key="ing_test", base_url="https://example.com")
    resp = client.cloud_submit_job(request=_request())

    assert resp.job_id == "job-1"
    assert resp.stage_id == "stage-1"
    assert resp.status == "queued"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/runs"
    assert captured["api_key"] == "ing_test"
    assert captured["base_url"] == "https://example.com"
    json_payload = captured["json_payload"]
    assert isinstance(json_payload, dict)
    assert json_payload["executor"] == {"type": "refiner-cloud"}


def test_cloud_client_cloud_submit_job_requires_job_and_stage_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "refiner.platform.client.request_json",
        lambda **_: {"status": "queued"},
    )

    client = MacrodataClient(api_key="ing_test", base_url="https://example.com")
    try:
        client.cloud_submit_job(request=_request())
    except MacrodataApiError as err:
        assert "job_id" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected MacrodataApiError")
