from __future__ import annotations

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
