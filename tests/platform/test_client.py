from __future__ import annotations

import pytest

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataClient


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


def test_client_raises_credentials_error_without_env_or_saved_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MACRODATA_API_KEY", raising=False)

    with pytest.raises(MacrodataCredentialsError, match="No credentials found") as exc:
        MacrodataClient()

    assert exc.value.missing is True


def test_client_prefers_env_api_key_over_saved_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MACRODATA_API_KEY", "md_env")

    client = MacrodataClient(base_url="https://example.com")

    assert client.api_key == "md_env"


def test_report_stage_heartbeat_posts_to_stage_heartbeat_route(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"stage": {"job_id": "job-1", "index": 2, "status": "running"}}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.report_stage_heartbeat(job_id="job-1", stage_index=2)

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/jobs/job-1/stages/2/heartbeat"
    assert captured["json_payload"] == {}
    assert response.stage.status == "running"


def test_cli_list_jobs_omits_me_when_false(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"items": [], "nextCursor": None}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cli_list_jobs(limit=20, me=False)

    assert captured["path"] == "/api/cli/jobs?limit=20"


def test_cli_list_jobs_serializes_me_as_lowercase_true(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"items": [], "nextCursor": None}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cli_list_jobs(limit=1, me=True)

    assert captured["path"] == "/api/cli/jobs?me=true&limit=1"


def test_cli_get_job_logs_serializes_anchor(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"entries": [], "hasOlder": False, "nextCursor": None}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cli_get_job_logs(
        job_id="job-1",
        start_ms=1,
        end_ms=2,
        anchor="earliest",
    )

    assert (
        captured["path"] == "/api/cli/jobs/job-1/logs?startMs=1&endMs=2&anchor=earliest"
    )


def test_cli_get_job_logs_omits_unset_bounds(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"entries": [], "hasOlder": False, "nextCursor": None}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cli_get_job_logs(
        job_id="job-1",
        anchor="latest",
    )

    assert captured["path"] == "/api/cli/jobs/job-1/logs?anchor=latest"
