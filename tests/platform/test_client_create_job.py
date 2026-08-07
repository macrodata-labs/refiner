from __future__ import annotations

from typing import cast

import pytest

from refiner.platform.client import MacrodataApiError, MacrodataClient


def _job_submit_response() -> dict[str, object]:
    return {
        "job": {
            "id": "job-1",
            "stages": [{"index": 0}],
        }
    }


def test_create_job_includes_manifest_refiner_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _job_submit_response()

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="ing_test", base_url="https://example.com")
    ctx = client.create_job(
        name="local job",
        plan={"stages": [{"name": "stage_0", "steps": []}]},
        manifest={
            "version": 1,
            "environment": {
                "refiner_version": "0.2.0",
                "refiner_ref": "abc123def456",
            },
        },
    )
    assert ctx.job_id == "job-1"
    assert ctx.stage_index == 0

    json_payload = cast(dict[str, object], captured["json_payload"])
    assert "executor" not in json_payload
    assert json_payload["manifest"] == {
        "version": 1,
        "environment": {
            "refiner_version": "0.2.0",
            "refiner_ref": "abc123def456",
        },
    }


def test_create_job_does_not_retry_request_timeout(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        del kwargs
        calls += 1
        raise MacrodataApiError(0, "read timed out")

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)
    monkeypatch.setattr("refiner.platform.client.api.time.sleep", sleeps.append)

    client = MacrodataClient(api_key="ing_test", base_url="https://example.com")
    with pytest.raises(MacrodataApiError, match="read timed out"):
        client.create_job(
            name="local job",
            plan={"stages": [{"name": "stage_0", "steps": []}]},
            manifest={"version": 1},
        )

    assert calls == 1
    assert sleeps == []
