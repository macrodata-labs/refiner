from __future__ import annotations

from typing import cast

from refiner.pipeline import read_jsonl
from refiner.platform.client import MacrodataClient


def _job_submit_response() -> dict[str, object]:
    return {
        "job": {
            "id": "job-1",
            "stages": [{"index": 0}],
        }
    }


def test_create_job_includes_manifest_refiner_ref(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _job_submit_response()

    monkeypatch.setattr("refiner.platform.client.request_json", fake_request_json)
    monkeypatch.setattr(
        "refiner.platform.client.compile_pipeline_plan",
        lambda _pipeline: {"stages": [{"name": "stage_0", "steps": []}]},
    )
    monkeypatch.setattr(
        "refiner.platform.client.build_run_manifest",
        lambda: {"version": 1, "environment": {"refiner_ref": "abc123def456"}},
    )

    client = MacrodataClient(api_key="ing_test", base_url="https://example.com")
    ctx = client.create_job(name="local job", pipeline=read_jsonl("input.jsonl"))
    assert ctx.job_id == "job-1"
    assert ctx.stage_id == "0"

    json_payload = cast(dict[str, object], captured["json_payload"])
    assert json_payload["executor"] == {"type": "refiner-local"}
    assert json_payload["manifest"] == {
        "version": 1,
        "environment": {"refiner_ref": "abc123def456"},
    }
