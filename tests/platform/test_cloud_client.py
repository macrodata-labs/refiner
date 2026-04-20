from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.platform.client import (
    CloudPipelinePayload,
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    StagePayload,
)
from refiner.platform.client import MacrodataApiError


def _request() -> CloudRunCreateRequest:
    return CloudRunCreateRequest(
        name="demo-cloud-job",
        plan={"stages": [{"name": "stage_0", "steps": []}]},
        stage_payloads=[
            StagePayload(
                stage_index=0,
                pipeline_payload=CloudPipelinePayload(
                    format="cloudpickle",
                    bytes_b64="AQID",
                    sha256="abc123",
                    size_bytes=3,
                ),
                runtime=CloudRuntimeConfig(
                    num_workers=2,
                    cpus_per_worker=4,
                    mem_mb_per_worker=8192,
                    gpus_per_worker=2,
                    gpu_type="h100",
                ),
            )
        ],
        secrets={"OPENAI_API_KEY": "test-secret"},
    )


def test_cloud_client_cloud_submit_job_posts_to_cloud_runs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "job_id": "job-1",
            "stage_index": 0,
            "status": "queued",
            "workspaceSlug": "macrodata",
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    resp = client.cloud_submit_job(request=_request())

    assert resp.job_id == "job-1"
    assert resp.stage_index == 0
    assert resp.status == "queued"
    assert resp.workspace_slug == "macrodata"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/runs"
    assert captured["base_url"] == "https://example.com"
    assert "http_client" in captured
    assert captured["timeout_s"] == 30.0
    json_payload = cast(dict[str, object], captured["json_payload"])
    assert json_payload["executor"] == {
        "type": "macrodata-cloud",
        "sync_local_dependencies": True,
    }
    stage_payloads = cast(list[dict[str, object]], json_payload["stage_payloads"])
    assert stage_payloads == [
        {
            "stage_index": 0,
            "pipeline_payload": {
                "format": "cloudpickle",
                "bytes_b64": "AQID",
                "sha256": "abc123",
                "size_bytes": 3,
            },
            "runtime": {
                "num_workers": 2,
                "cpus_per_worker": 4,
                "mem_mb_per_worker": 8192,
                "gpus_per_worker": 2,
                "gpu_type": "h100",
            },
        }
    ]
    assert json_payload["secrets"] == {"OPENAI_API_KEY": "test-secret"}


def test_cloud_client_cloud_submit_job_requires_job_and_stage_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "refiner.platform.client.api.request_json",
        lambda **_: {"status": "queued"},
    )

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    try:
        client.cloud_submit_job(request=_request())
    except MacrodataApiError as err:
        assert "job_id" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected MacrodataApiError")


def test_cloud_client_cloud_submit_job_posts_continue_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "job_id": "job-2",
            "stage_index": 1,
            "status": "queued",
            "workspaceSlug": "macrodata",
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    resp = client.cloud_submit_job(
        request=CloudRunCreateRequest(
            name="demo-cloud-job",
            continue_from_job="job-previous:2",
            plan={"stages": [{"name": "stage_0", "steps": []}]},
            stage_payloads=[
                StagePayload(
                    stage_index=0,
                    pipeline_payload=CloudPipelinePayload(
                        format="cloudpickle",
                        bytes_b64="AQID",
                        sha256="abc123",
                        size_bytes=3,
                    ),
                )
            ],
            manifest={"version": 1},
            sync_local_dependencies=False,
            runtime_overrides={
                "cpus_per_worker": 8,
                "mem_mb_per_worker": 16384,
            },
            force_continue=True,
        )
    )

    assert resp.job_id == "job-2"
    assert resp.stage_index == 1
    assert resp.status == "queued"
    assert resp.workspace_slug == "macrodata"
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/runs"
    assert captured["timeout_s"] == 30.0
    json_payload = cast(dict[str, object], captured["json_payload"])
    assert json_payload["name"] == "demo-cloud-job"
    assert json_payload["continue_from_job"] == "job-previous:2"
    assert json_payload["executor"] == {
        "type": "macrodata-cloud",
        "sync_local_dependencies": False,
    }
    assert json_payload["force_continue"] is True
    assert json_payload["runtime_overrides"] == {
        "cpus_per_worker": 8,
        "mem_mb_per_worker": 16384,
    }
    assert json_payload["plan"] == {"stages": [{"name": "stage_0", "steps": []}]}
    assert json_payload["manifest"] == {"version": 1}
    assert json_payload["stage_payloads"] == [
        {
            "stage_index": 0,
            "pipeline_payload": {
                "format": "cloudpickle",
                "bytes_b64": "AQID",
                "sha256": "abc123",
                "size_bytes": 3,
            },
        }
    ]
