from __future__ import annotations

import httpx
import pytest

from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError, MacrodataClient


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


def test_lifecycle_request_retries_transient_errors_with_backoff(monkeypatch) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["path"]))
        if len(calls) < 4:
            raise MacrodataApiError(0, "The read operation timed out")
        return {"stage": {"job_id": "job-1", "index": 2, "status": "running"}}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)
    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.report_stage_heartbeat(job_id="job-1", stage_index=2)

    assert response.stage.status == "running"
    assert calls == [
        "/api/jobs/job-1/stages/2/heartbeat",
        "/api/jobs/job-1/stages/2/heartbeat",
        "/api/jobs/job-1/stages/2/heartbeat",
        "/api/jobs/job-1/stages/2/heartbeat",
    ]
    assert sleeps == [0.25, 0.5, 1.0]


def test_lifecycle_request_retries_httpx_timeouts(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_http_request(method: str, url: str, **kwargs: object) -> httpx.Response:
        nonlocal calls
        del kwargs
        calls += 1
        if calls < 4:
            raise httpx.ReadTimeout("The read operation timed out")
        return httpx.Response(
            200,
            json={"stage": {"job_id": "job-1", "index": 2, "status": "running"}},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    monkeypatch.setattr(client._http_client, "request", fake_http_request)

    response = client.report_stage_heartbeat(job_id="job-1", stage_index=2)

    assert response.stage.status == "running"
    assert calls == 4
    assert sleeps == [0.25, 0.5, 1.0]


def test_lifecycle_request_timeout_error_reports_url(monkeypatch) -> None:
    sleeps: list[float] = []

    def fake_http_request(method: str, url: str, **kwargs: object) -> httpx.Response:
        del method, url, kwargs
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    monkeypatch.setattr(client._http_client, "request", fake_http_request)

    with pytest.raises(MacrodataApiError) as exc:
        client.report_stage_heartbeat(job_id="job-1", stage_index=2)

    assert exc.value.status == 0
    assert exc.value.url == "https://example.com/api/jobs/job-1/stages/2/heartbeat"
    assert (
        str(exc.value) == "HTTP 0: The read operation timed out "
        "(url: https://example.com/api/jobs/job-1/stages/2/heartbeat)"
    )
    assert sleeps == [0.25, 0.5, 1.0]


def test_lifecycle_request_does_not_retry_conflicts(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        del kwargs
        calls += 1
        raise MacrodataApiError(409, "Worker already finished")

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)
    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    with pytest.raises(MacrodataApiError, match="Worker already finished"):
        client.report_stage_finished(job_id="job-1", stage_index=2, status="failed")

    assert calls == 1
    assert sleeps == []


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


def test_cli_delete_secret_encodes_name_and_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    client.cli_delete_secret(name="HF/TOKEN", env="production")

    assert captured["method"] == "DELETE"
    assert captured["path"] == "/api/cli/secrets/HF%2FTOKEN?env=production"


@pytest.mark.parametrize(
    ("operation", "expected_path"),
    [
        (
            lambda client: client.cli_cancel_job(job_id="job-1"),
            "/api/cli/jobs/job-1/cancel",
        ),
        (
            lambda client: client.cli_set_secret(
                name="HF_TOKEN",
                value="secret",
                env="production",
            ),
            "/api/cli/secrets",
        ),
        (
            lambda client: client.cli_delete_secret(
                name="HF_TOKEN",
                env="production",
            ),
            "/api/cli/secrets/HF_TOKEN?env=production",
        ),
    ],
)
def test_cli_mutations_retry_transient_failures(
    monkeypatch,
    operation,
    expected_path: str,
) -> None:
    calls = 0
    captured: dict[str, object] = {}
    sleeps: list[float] = []

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        captured.update(kwargs)
        if calls == 1:
            raise MacrodataApiError(503, "try again")
        return {"success": True}

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)
    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    assert operation(client) == {"success": True}

    assert calls == 2
    assert captured["path"] == expected_path
    assert sleeps == [0.25]
