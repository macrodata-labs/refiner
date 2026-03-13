from __future__ import annotations

from refiner.platform.client import MacrodataClient


def test_shard_finalized_workers_returns_typed_rows(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_request_json(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "shards": [
                {"shard_id": "shard-1", "worker_id": "worker-a"},
                {"shard_id": "shard-2", "worker_id": "worker-b"},
            ]
        }

    monkeypatch.setattr("refiner.platform.client.api.request_json", fake_request_json)

    client = MacrodataClient(api_key="md_test", base_url="https://example.com")
    response = client.shard_finalized_workers(job_id="job-1", stage_index=0)

    assert captured["method"] == "GET"
    assert captured["path"] == "/api/jobs/job-1/stages/0/shards/finalized-workers"
    assert response.shards[0].shard_id == "shard-1"
    assert response.shards[0].worker_id == "worker-a"
    assert response.shards[1].shard_id == "shard-2"
    assert response.shards[1].worker_id == "worker-b"
