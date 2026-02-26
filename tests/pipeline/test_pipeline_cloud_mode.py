from __future__ import annotations

from refiner.pipeline import read_jsonl
from refiner.platform.cloud.models import CloudPipelinePayload


def test_pipeline_launch_cloud_submits_compiled_plan(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self, *, api_key: str, base_url: str | None = None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_id = "stage-456"
                status = "queued"

            return _Resp()

    monkeypatch.setattr("refiner.runtime.launchers.cloud.current_api_key", lambda: "ing_test")
    monkeypatch.setattr(
        "refiner.runtime.launchers.cloud.MacrodataClient", FakeMacrodataClient
    )
    monkeypatch.setattr(
        "refiner.runtime.launchers.cloud.serialize_pipeline_inline",
        lambda pipeline: CloudPipelinePayload(
            format="cloudpickle",
            bytes_b64="AQID",
            sha256="abc123",
            size_bytes=3,
        ),
    )
    monkeypatch.setattr(
        "refiner.runtime.launchers.cloud.compile_pipeline_plan",
        lambda pipeline: {"stages": [{"name": "stage_0", "steps": []}]},
    )

    pipeline = read_jsonl("input.jsonl")
    result = pipeline.launch_cloud(
        name="demo cloud",
        num_workers=3,
        heartbeat_every_rows=2048,
    )

    assert result.job_id == "job-123"
    assert result.stage_id == "stage-456"
    assert result.status == "queued"
    assert captured["api_key"] == "ing_test"
    assert captured["base_url"] is None

    request = captured["request"]
    assert request.name == "demo cloud"
    assert request.runtime.num_workers == 3
    assert request.runtime.heartbeat_every_rows == 2048
    assert request.plan["stages"][0]["name"] == "stage_0"
