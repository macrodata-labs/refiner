from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import httpx
import msgspec

from refiner.pipeline.resources import GPU
from refiner.platform.client import (
    CloudFile,
    CloudFileCompleteRequestItem,
    CloudFileUploadInstruction,
    CloudFileUploadRequestItem,
    CloudFileUploadStatus,
    CloudRunCreateRequest,
    CloudRuntimeConfig,
    MacrodataApiError,
    MacrodataClient,
    StagePayload,
)
from refiner.services import RuntimeServiceSpec

_TEST_TIMESTAMP = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)


def _request() -> CloudRunCreateRequest:
    return CloudRunCreateRequest(
        name="demo-cloud-job",
        plan={"stages": [{"name": "stage_0", "steps": []}]},
        stage_payloads=[
            StagePayload(
                stage_index=0,
                pipeline_payload=CloudFile(
                    file_id="00000000-0000-7000-8000-000000000123",
                ),
                runtime=CloudRuntimeConfig(
                    num_workers=2,
                    cpus_per_worker=4,
                    mem_mb_per_worker=8192,
                    gpu=GPU(count=2, type="h100", cuda_version="12.4"),
                ),
                runtime_services=(
                    RuntimeServiceSpec(
                        name="vllm-demo",
                        kind="llm",
                        config={
                            "model_name_or_path": "Qwen/Qwen3.5-9B",
                            "config": "throughput",
                        },
                    ),
                ),
            )
        ],
        secrets=[{"OPENAI_API_KEY": "test-secret"}],
        env={"MODEL_NAME": "gpt-5"},
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
            "warnings": ["warning 1"],
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    resp = client.cloud_submit_job(request=_request())

    assert resp.job_id == "job-1"
    assert resp.stage_index == 0
    assert resp.status == "queued"
    assert resp.workspace_slug == "macrodata"
    assert resp.warnings == ["warning 1"]
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/runs"
    assert captured["base_url"] == "https://example.com"
    assert "http_client" in captured
    assert captured["timeout_s"] == 30.0
    json_payload = cast(dict[str, object], captured["json_payload"])
    assert json_payload["executor"] == {}
    stage_payloads = cast(list[dict[str, object]], json_payload["stage_payloads"])
    assert stage_payloads == [
        {
            "stage_index": 0,
            "pipeline_payload": {
                "file_id": "00000000-0000-7000-8000-000000000123",
            },
            "runtime": {
                "num_workers": 2,
                "cpus_per_worker": 4,
                "mem_mb_per_worker": 8192,
                "gpu": {
                    "count": 2,
                    "type": "h100",
                    "cuda_version": "12.4",
                },
            },
            "runtime_services": [
                {
                    "name": "vllm-demo",
                    "kind": "llm",
                    "config": {
                        "model_name_or_path": "Qwen/Qwen3.5-9B",
                        "config": "throughput",
                    },
                }
            ],
        }
    ]
    assert json_payload["secrets"] == [{"OPENAI_API_KEY": "test-secret"}]
    assert json_payload["env"] == {"MODEL_NAME": "gpt-5"}


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
            "warnings": ["warning 2"],
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    resp = client.cloud_submit_job(
        request=CloudRunCreateRequest(
            name="demo-cloud-job",
            continue_from_job="00000000-0000-1000-8000-000000000123:2",
            plan={"stages": [{"name": "stage_0", "steps": []}]},
            stage_payloads=[
                StagePayload(
                    stage_index=0,
                    pipeline_payload=CloudFile(
                        file_id="00000000-0000-7000-8000-000000000123",
                    ),
                    runtime=CloudRuntimeConfig(num_workers=1),
                )
            ],
            manifest={"version": 1},
            unsafe_continue=True,
        )
    )

    assert resp.job_id == "job-2"
    assert resp.stage_index == 1
    assert resp.status == "queued"
    assert resp.workspace_slug == "macrodata"
    assert resp.warnings == ["warning 2"]
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/runs"
    assert captured["timeout_s"] == 30.0
    json_payload = cast(dict[str, object], captured["json_payload"])
    assert json_payload["name"] == "demo-cloud-job"
    assert json_payload["continue_from_job"] == "00000000-0000-1000-8000-000000000123:2"
    assert json_payload["executor"] == {}
    assert json_payload["unsafe_continue"] is True
    assert json_payload["plan"] == {"stages": [{"name": "stage_0", "steps": []}]}
    assert json_payload["manifest"] == {"version": 1}
    assert json_payload["stage_payloads"] == [
        {
            "stage_index": 0,
            "pipeline_payload": {
                "file_id": "00000000-0000-7000-8000-000000000123",
            },
            "runtime": {"num_workers": 1},
        }
    ]


def test_cloud_client_creates_file_upload_urls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "files": [
                {
                    "file_id": "00000000-0000-7000-8000-000000000123",
                    "sha256": "a" * 64,
                    "size_bytes": 3,
                    "status": "new",
                    "url": "https://payloads.example/upload",
                    "required_headers": {
                        "content-length": "3",
                        "x-amz-checksum-sha256": "checksum",
                    },
                    "upload_url_expires_at": "2026-05-20T12:00:00Z",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.cloud_create_file_upload_urls(
        files=[
            CloudFileUploadRequestItem(
                sha256="a" * 64,
                size_bytes=3,
            )
        ],
        object_ttl_secs=None,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/files/upload-urls"
    assert captured["base_url"] == "https://example.com"
    assert captured["timeout_s"] == 30.0
    assert captured["json_payload"] == {
        "files": [{"sha256": "a" * 64, "size_bytes": 3}],
        "object_ttl_secs": None,
    }
    assert response.files[0].file_id == "00000000-0000-7000-8000-000000000123"
    assert response.files[0].status is CloudFileUploadStatus.NEW
    assert response.files[0].upload_url_expires_at == _TEST_TIMESTAMP
    assert response.files[0].expires_at is None
    assert response.files[0].required_headers is not None
    assert response.files[0].required_headers["x-amz-checksum-sha256"] == "checksum"


def test_cloud_file_upload_status_serializes_as_wire_literal() -> None:
    assert msgspec.json.encode(CloudFileUploadStatus.NEW) == b'"new"'
    assert msgspec.json.encode(CloudFileUploadStatus.EXISTS) == b'"exists"'


def test_cloud_client_uploads_cloud_file_without_macrodata_auth(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 204

    def fake_httpx_request(**kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("refiner.platform.client.api.httpx.request", fake_httpx_request)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cloud_upload_file(
        instruction=CloudFileUploadInstruction(
            file_id="00000000-0000-7000-8000-000000000123",
            sha256="a" * 64,
            size_bytes=7,
            status=CloudFileUploadStatus.NEW,
            url="https://payloads.example/upload",
            required_headers={
                "content-length": "7",
                "x-amz-checksum-sha256": "checksum",
            },
            upload_url_expires_at=_TEST_TIMESTAMP,
            expires_at=None,
        ),
        payload_bytes=b"payload",
    )

    assert captured["method"] == "PUT"
    assert captured["url"] == "https://payloads.example/upload"
    assert captured["content"] == b"payload"
    headers = cast(dict[str, str], captured["headers"])
    assert headers["content-length"] == "7"
    assert headers["x-amz-checksum-sha256"] == "checksum"
    assert "Authorization" not in headers
    assert "User-Agent" not in headers


def test_cloud_client_cloud_upload_file_noops_when_file_exists(monkeypatch) -> None:
    def fake_httpx_request(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("existing cloud files should not be uploaded")

    monkeypatch.setattr("refiner.platform.client.api.httpx.request", fake_httpx_request)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cloud_upload_file(
        instruction=CloudFileUploadInstruction(
            file_id="00000000-0000-7000-8000-000000000123",
            sha256="a" * 64,
            size_bytes=7,
            status=CloudFileUploadStatus.EXISTS,
            expires_at=None,
        ),
        payload_bytes=b"payload",
    )


def test_cloud_client_cloud_file_upload_failure_includes_provider_message(
    monkeypatch,
) -> None:
    def fake_httpx_request(**kwargs: object) -> httpx.Response:
        del kwargs
        return httpx.Response(
            403,
            json={"message": "SignatureDoesNotMatch"},
        )

    monkeypatch.setattr("refiner.platform.client.api.httpx.request", fake_httpx_request)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    try:
        client.cloud_upload_file(
            instruction=CloudFileUploadInstruction(
                file_id="00000000-0000-7000-8000-000000000123",
                sha256="a" * 64,
                size_bytes=7,
                status=CloudFileUploadStatus.NEW,
                url="https://payloads.example/upload",
                required_headers={"content-length": "7"},
            ),
            payload_bytes=b"payload",
        )
    except MacrodataApiError as err:
        assert err.status == 403
        assert err.message == "Failed to upload cloud file: SignatureDoesNotMatch"
    else:  # pragma: no cover
        raise AssertionError("expected upload failure")


def test_cloud_client_completes_cloud_files(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "files": [
                {
                    "file_id": "00000000-0000-7000-8000-000000000123",
                    "sha256": "a" * 64,
                    "size_bytes": 3,
                    "uploaded_at": "2026-05-20T12:00:00Z",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.cloud_complete_files(
        files=[
            CloudFileCompleteRequestItem(file_id="00000000-0000-7000-8000-000000000123")
        ],
        object_ttl_secs=None,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/cloud/files/complete"
    assert captured["json_payload"] == {
        "files": [{"file_id": "00000000-0000-7000-8000-000000000123"}],
        "object_ttl_secs": None,
    }
    assert response.files[0].file_id == "00000000-0000-7000-8000-000000000123"
    assert response.files[0].uploaded_at == _TEST_TIMESTAMP
    assert response.files[0].expires_at is None
