from __future__ import annotations

from typing import cast

from refiner.pipeline import read_jsonl
from refiner.pipeline.data.shard import Shard
from refiner.platform.client import (
    CloudPipelinePayload,
    CloudRunCreateRequest,
    ShardDescriptor,
)


def test_pipeline_launch_cloud_submits_compiled_plan(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self):
            captured["client_initialized"] = True
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_id = "stage-456"
                status = "queued"

            return _Resp()

    monkeypatch.setattr("refiner.launchers.base.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.serialize_pipeline_inline",
        lambda pipeline: CloudPipelinePayload(
            format="cloudpickle",
            bytes_b64="AQID",
            sha256="abc123",
            size_bytes=3,
        ),
    )
    monkeypatch.setattr(
        "refiner.launchers.base.compile_pipeline_plan",
        lambda pipeline: {"stages": [{"name": "stage_0", "steps": []}]},
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.compile_shard_descriptors",
        lambda shards: [ShardDescriptor.from_shard(s) for s in shards],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda: {
            "version": 1,
            "environment": {"refiner_ref": "abc123def456"},
            "script": {"text": "print('hi')"},
        },
    )

    pipeline = read_jsonl("input.jsonl")
    monkeypatch.setattr(
        pipeline.source,
        "list_shards",
        lambda: [Shard(path="input.jsonl", start=0, end=1)],
    )
    result = pipeline.launch_cloud(
        name="demo cloud",
        num_workers=3,
        heartbeat_interval_seconds=12,
        cpus_per_worker=2,
        mem_mb_per_worker=8192,
    )

    assert result.job_id == "job-123"
    assert result.stage_id == "stage-456"
    assert result.status == "queued"
    assert captured["client_initialized"] is True

    request = cast(CloudRunCreateRequest, captured["request"])
    assert request.name == "demo cloud"
    assert request.runtime.num_workers == 3
    assert request.runtime.heartbeat_interval_seconds == 12
    assert request.runtime.cpus_per_worker == 2
    assert request.runtime.mem_mb_per_worker == 8192
    assert request.sync_local_dependencies is True
    assert request.shards[0].path == "input.jsonl"
    assert request.plan["stages"][0]["name"] == "stage_0"
    assert request.manifest == {
        "version": 1,
        "environment": {"refiner_ref": "abc123def456"},
        "script": {"text": "print('hi')"},
    }


def test_pipeline_launch_cloud_can_disable_dependency_install(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_id = "stage-456"
                status = "queued"

            return _Resp()

    monkeypatch.setattr("refiner.launchers.base.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.serialize_pipeline_inline",
        lambda pipeline: CloudPipelinePayload(
            format="cloudpickle",
            bytes_b64="AQID",
            sha256="abc123",
            size_bytes=3,
        ),
    )
    monkeypatch.setattr(
        "refiner.launchers.base.compile_pipeline_plan",
        lambda pipeline: {"stages": [{"name": "stage_0", "steps": []}]},
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.compile_shard_descriptors",
        lambda shards: [ShardDescriptor.from_shard(s) for s in shards],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda: {"version": 1},
    )

    pipeline = read_jsonl("input.jsonl")
    monkeypatch.setattr(
        pipeline.source,
        "list_shards",
        lambda: [Shard(path="input.jsonl", start=0, end=1)],
    )
    pipeline.launch_cloud(name="demo cloud", sync_local_dependencies=False)

    request = cast(CloudRunCreateRequest, captured["request"])
    assert request.sync_local_dependencies is False
