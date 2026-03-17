from __future__ import annotations

from typing import cast

from refiner.pipeline import read_jsonl
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.platform.client import (
    CloudPipelinePayload,
    CloudRunCreateRequest,
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
                stage_index = 0
                status = "queued"
                workspace_slug = None

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
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            )
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {
            "version": 1,
            "environment": {"refiner_ref": "abc123def456"},
            "script": {"text": "print('hi')"},
        },
    )

    pipeline = read_jsonl("input.jsonl")
    result = pipeline.launch_cloud(
        name="demo cloud",
        num_workers=3,
        heartbeat_interval_seconds=12,
        cpus_per_worker=2,
        mem_mb_per_worker=8192,
    )

    assert result.job_id == "job-123"
    assert result.stage_index == 0
    assert result.status == "queued"
    assert captured["client_initialized"] is True

    request = cast(CloudRunCreateRequest, captured["request"])
    assert request.name == "demo cloud"
    assert request.sync_local_dependencies is True
    assert request.plan["stages"][0]["name"] == "stage_0"
    assert request.plan["stages"][0]["requested_num_workers"] == 3
    assert len(request.stage_payloads) == 1
    assert request.stage_payloads[0].stage_index == 0
    assert request.stage_payloads[0].pipeline_payload.sha256 == "abc123"
    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.heartbeat_interval_seconds == 12
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
    assert request.stage_payloads[0].runtime.mem_mb_per_worker == 8192
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
                stage_index = 0
                status = "queued"
                workspace_slug = None

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
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            )
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {"version": 1},
    )

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(name="demo cloud", sync_local_dependencies=False)

    request = cast(CloudRunCreateRequest, captured["request"])
    assert request.sync_local_dependencies is False


def test_pipeline_launch_cloud_resolves_secrets(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 0
                status = "queued"
                workspace_slug = None

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
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            )
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {"version": 1},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(
        name="demo cloud",
        secrets={"OPENAI_API_KEY": None, "MODEL_NAME": "gpt-5"},
    )

    request = cast(CloudRunCreateRequest, captured["request"])
    assert request.secrets == {
        "OPENAI_API_KEY": "env-secret",
        "MODEL_NAME": "gpt-5",
    }


def test_pipeline_launch_cloud_redacts_captured_strings_in_outgoing_request(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    secret = "super-secret-value"

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 0
                status = "queued"
                workspace_slug = None

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
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            )
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda: {
            "version": 1,
            "script": {"text": "API_KEY = 'super-secret-value'"},
        },
    )

    pipeline = read_jsonl("input.jsonl").map(
        lambda row: {"token": "super-secret-value", "x": row["x"]}
    )
    pipeline.launch_cloud(name="demo cloud", secrets={"OPENAI_API_KEY": secret})

    request = cast(CloudRunCreateRequest, captured["request"])
    assert "REDACTED_SECRET" in request.plan["stages"][0]["steps"][1]["args"]["fn"]
    assert secret not in request.plan["stages"][0]["steps"][1]["args"]["fn"]
    assert request.manifest is not None
    assert "REDACTED_SECRET" in request.manifest["script"]["text"]
    assert secret not in request.manifest["script"]["text"]


def test_pipeline_launch_cloud_requires_missing_env_secret(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):  # pragma: no cover
            raise AssertionError("cloud_submit_job should not be called")

    monkeypatch.setattr("refiner.launchers.base.MacrodataClient", FakeMacrodataClient)
    pipeline = read_jsonl("input.jsonl")

    try:
        pipeline.launch_cloud(name="demo cloud", secrets={"OPENAI_API_KEY": None})
    except SystemExit as err:
        assert "OPENAI_API_KEY" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_pipeline_launch_cloud_requires_platform_auth_before_secret_resolution(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "refiner.launchers.base.BaseLauncher._platform_client_or_none",
        lambda self: None,
    )
    pipeline = read_jsonl("input.jsonl")

    try:
        pipeline.launch_cloud(name="demo cloud", secrets={"OPENAI_API_KEY": None})
    except SystemExit as err:
        assert "MACRODATA_API_KEY" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_pipeline_launch_cloud_submits_one_stage_payload_per_planned_stage(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            captured["request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 0
                status = "queued"
                workspace_slug = None

            return _Resp()

    monkeypatch.setattr("refiner.launchers.base.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.serialize_pipeline_inline",
        lambda pipeline: CloudPipelinePayload(
            format="cloudpickle",
            bytes_b64=f"payload-{id(pipeline)}",
            sha256=f"sha-{id(pipeline)}",
            size_bytes=3,
        ),
    )
    first_stage_pipeline = read_jsonl("input-a.jsonl")
    second_stage_pipeline = read_jsonl("input-b.jsonl")
    monkeypatch.setattr(
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=first_stage_pipeline,
                compute=StageComputeRequirements(num_workers=2),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=second_stage_pipeline,
                compute=StageComputeRequirements(num_workers=5),
            ),
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {"version": 1},
    )

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(name="demo cloud")

    request = cast(CloudRunCreateRequest, captured["request"])
    assert [payload.stage_index for payload in request.stage_payloads] == [0, 1]
    assert [payload.runtime.num_workers for payload in request.stage_payloads] == [2, 5]
    assert (
        request.stage_payloads[0].pipeline_payload.sha256
        != request.stage_payloads[1].pipeline_payload.sha256
    )
