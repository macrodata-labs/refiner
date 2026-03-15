from __future__ import annotations

from typing import Any, cast

from refiner.platform.client import OkResponse, RunHandle, ShardClaimResponse
from refiner.platform.client import SerializedShard
from refiner.worker.lifecycle.platform import PlatformRuntimeLifecycle
from refiner.pipeline.data.shard import FilePart, Shard


def test_platform_runtime_register_and_lifecycle() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeClient:
        def shard_register(self, **kwargs):
            calls.append(("register", kwargs))
            return OkResponse()

        def shard_claim(self, **kwargs):
            calls.append(("claim", kwargs))
            return ShardClaimResponse(
                shard=SerializedShard(
                    shard_id=Shard.from_file_parts(
                        [FilePart(path="p0", start=0, end=1)]
                    ).id,
                    descriptor={
                        "kind": "file_parts",
                        "parts": [
                            {
                                "path": "p0",
                                "start": 0,
                                "end": 1,
                                "source_index": 0,
                                "unit": "bytes",
                            }
                        ],
                    },
                )
            )

        def shard_heartbeat(self, **kwargs):
            calls.append(("heartbeat", kwargs))
            return OkResponse()

        def shard_finish(self, **kwargs):
            calls.append(("finish", kwargs))
            return OkResponse()

    lifecycle = PlatformRuntimeLifecycle(
        run=RunHandle(
            job_id="job-1",
            stage_index=1,
            client=cast(Any, FakeClient()),
            worker_id="worker-7",
        ),
    )
    shards = [Shard.from_file_parts([FilePart(path="p0", start=0, end=1)])]
    lifecycle.seed_shards(shards)
    claimed = lifecycle.claim()
    assert claimed is not None
    lifecycle.heartbeat([claimed])
    lifecycle.complete(claimed)
    lifecycle.fail(claimed, "boom")

    assert [name for name, _ in calls] == [
        "register",
        "claim",
        "heartbeat",
        "finish",
        "finish",
    ]
    register_kwargs = calls[0][1]
    assert register_kwargs["job_id"] == "job-1"
    assert register_kwargs["stage_index"] == 1


def test_platform_runtime_claim_none_when_queue_empty() -> None:
    class FakeClient:
        def shard_claim(self, **kwargs):
            return ShardClaimResponse(shard=None)

    lifecycle = PlatformRuntimeLifecycle(
        run=RunHandle(
            job_id="job-1",
            stage_index=1,
            client=cast(Any, FakeClient()),
            worker_id="worker-1",
        ),
    )
    assert lifecycle.claim() is None
