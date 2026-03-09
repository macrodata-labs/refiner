from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline


def test_create_job_treats_whitespace_workspace_slug_as_none(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "refiner.platform.client.compile_pipeline_plan",
        lambda pipeline: [{"stage": "s1"}],
    )
    monkeypatch.setattr(
        "refiner.platform.client.request_json",
        lambda **_: {
            "job": {
                "id": "job-1",
                "stages": [{"index": 0}],
                "workspaceSlug": "   ",
            }
        },
    )

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    context = client.create_job(name="Job", pipeline=cast(RefinerPipeline, object()))

    assert context.job_id == "job-1"
    assert context.stage_id == "0"
    assert context.workspace_slug is None
