from __future__ import annotations

from refiner.ledger import CloudLedger
from refiner.ledger.shard import Shard


def test_cloud_ledger_register_and_lifecycle(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str | None = None):
            assert api_key == "ing_test"
            assert base_url == "https://example.com"

        def cloud_ledger_register_stage_shards(self, **kwargs):
            calls.append(("register", kwargs))
            return {"ok": True}

        def cloud_ledger_claim_shard(self, **kwargs):
            calls.append(("claim", kwargs))
            return {"shard": {"path": "p0", "start": 0, "end": 1}}

        def cloud_ledger_heartbeat_shard(self, **kwargs):
            calls.append(("heartbeat", kwargs))
            return {"ok": True}

        def cloud_ledger_complete_shard(self, **kwargs):
            calls.append(("complete", kwargs))
            return {"ok": True}

        def cloud_ledger_fail_shard(self, **kwargs):
            calls.append(("fail", kwargs))
            return {"ok": True}

    monkeypatch.setattr("refiner.ledger.backend.cloud.MacrodataClient", FakeClient)

    ledger = CloudLedger(
        run_id="run-1",
        worker_id=7,
        job_id="job-1",
        stage_id="stage-1",
        api_key="ing_test",
        base_url="https://example.com",
    )
    shards = [Shard(path="p0", start=0, end=1)]
    ledger.seed_shards(shards)
    claimed = ledger.claim()
    assert claimed is not None
    ledger.heartbeat(claimed)
    ledger.complete(claimed)
    ledger.fail(claimed, "boom")

    assert [name for name, _ in calls] == [
        "register",
        "claim",
        "heartbeat",
        "complete",
        "fail",
    ]
    register_kwargs = calls[0][1]
    assert register_kwargs["job_id"] == "job-1"
    assert register_kwargs["stage_id"] == "stage-1"


def test_cloud_ledger_claim_none_when_queue_empty(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str | None = None):
            pass

        def cloud_ledger_claim_shard(self, **kwargs):
            return {"shard": None}

    monkeypatch.setattr("refiner.ledger.backend.cloud.MacrodataClient", FakeClient)

    ledger = CloudLedger(
        run_id="run-1",
        worker_id=1,
        job_id="job-1",
        stage_id="stage-1",
        api_key="ing_test",
    )
    assert ledger.claim() is None

