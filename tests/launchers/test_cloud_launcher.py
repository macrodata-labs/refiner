from __future__ import annotations

import pytest
from collections.abc import Callable
from typing import cast

from refiner.pipeline import read_jsonl
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudPipelinePayload,
    CloudRunCreateRequest,
    CloudRunResumeRequest,
    MacrodataApiError,
)
from refiner.platform.manifest import _redact_captured_text


def _stub_cloud_submit(
    monkeypatch,
    *,
    manifest: dict[str, object] | Callable[..., dict[str, object]] | None = None,
    fail_on_submit: bool = False,
    fail_on_resume: bool = False,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):
            if fail_on_submit:
                raise AssertionError("should not submit")
            captured["submit_request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 0
                status = "queued"
                workspace_slug = None

            return _Resp()

        def cloud_resume_job(self, *, request):
            if fail_on_resume:
                raise AssertionError("should not resume")
            captured["resume_request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 1
                status = "queued"
                workspace_slug = None

            return _Resp()

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
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
        manifest if callable(manifest) else (lambda **_: manifest or {"version": 1}),
    )
    return captured


def test_pipeline_launch_cloud_submits_compiled_plan(monkeypatch) -> None:
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest={
            "version": 1,
            "environment": {"refiner_version": "0.2.0", "refiner_ref": "abc123def456"},
            "script": {"text": "print('hi')"},
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl")
    result = pipeline.launch_cloud(
        name="demo cloud",
        num_workers=3,
        cpus_per_worker=2,
        mem_mb_per_worker=4096,
        gpus_per_worker=2,
        gpu_type="h100",
    )

    assert result.job_id == "job-123"
    assert result.stage_index == 0
    assert result.status == "queued"

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.name == "demo cloud"
    assert request.sync_local_dependencies is True
    assert request.plan["stages"][0]["name"] == "stage_0"
    assert request.plan["stages"][0]["requested_num_workers"] == 3
    assert request.plan["stages"][0]["cpus_per_worker"] == 2
    assert request.plan["stages"][0]["memory_mb_per_worker"] == 4096
    assert request.plan["stages"][0]["gpus_per_worker"] == 2
    assert request.plan["stages"][0]["gpu_type"] == "h100"
    assert len(request.stage_payloads) == 1
    assert request.stage_payloads[0].stage_index == 0
    assert request.stage_payloads[0].pipeline_payload.sha256 == "abc123"
    assert request.stage_payloads[0].runtime is not None
    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
    assert request.stage_payloads[0].runtime.mem_mb_per_worker == 4096
    assert request.stage_payloads[0].runtime.gpus_per_worker == 2
    assert request.stage_payloads[0].runtime.gpu_type == "h100"
    assert request.manifest == {
        "version": 1,
        "environment": {"refiner_version": "0.2.0", "refiner_ref": "abc123def456"},
        "script": {"text": "print('hi')"},
    }


def test_pipeline_launch_cloud_requires_gpu_type_with_gpu_count(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError,
        match="gpu_type is required when gpus_per_worker is set",
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            gpus_per_worker=2,
        )


def test_pipeline_launch_cloud_requires_gpu_count_with_gpu_type(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError,
        match="gpus_per_worker is required when gpu_type is set",
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            gpu_type="h100",
        )


def test_pipeline_launch_cloud_rejects_non_positive_gpu_count(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="gpus_per_worker must be > 0"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            gpus_per_worker=0,
            gpu_type="h100",
        )


def test_pipeline_launch_cloud_rejects_blank_gpu_type(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="gpu_type must be non-empty"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            gpus_per_worker=1,
            gpu_type="   ",
        )


def test_pipeline_launch_cloud_can_disable_dependency_install(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(name="demo cloud", sync_local_dependencies=False)

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.sync_local_dependencies is False


def test_pipeline_launch_cloud_resolves_secrets(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(
        name="demo cloud",
        secrets={"OPENAI_API_KEY": None, "MODEL_NAME": "gpt-5"},
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == {
        "OPENAI_API_KEY": "env-secret",
        "MODEL_NAME": "gpt-5",
    }


def test_pipeline_launch_cloud_sends_env_without_redacting_it(monkeypatch) -> None:
    secret = "super-secret-value"
    env_value = "plain-env-value"
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest={"version": 1, "script": {"text": "TOKEN='super-secret-value'"}},
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl").map(
        lambda row: {"token": "super-secret-value", "x": row["x"]}
    )
    pipeline.launch_cloud(
        name="demo cloud",
        secrets={"OPENAI_API_KEY": secret},
        env={"MODEL_NAME": env_value},
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == {
        "OPENAI_API_KEY": "super-secret-value",
        "MODEL_NAME": "plain-env-value",
    }
    assert "REDACTED_SECRET" in request.plan["stages"][0]["steps"][1]["args"]["fn"]


def test_pipeline_launch_cloud_rejects_overlapping_secret_and_env_keys(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(SystemExit, match="API_KEY"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            secrets={"API_KEY": "secret"},
            env={"API_KEY": "env"},
        )


def test_pipeline_launch_cloud_redacts_captured_strings_in_outgoing_request(
    monkeypatch,
) -> None:
    secret = "super-secret-value"
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest=lambda **kwargs: {
            "version": 1,
            "script": {
                "path": "/tmp/super-secret-value_job.py",
                "text": _redact_captured_text(
                    "API_KEY = 'super-secret-value'",
                    secret_values=kwargs.get("secret_values", ()),
                ),
            },
            "environment": {
                "refiner_version": "0.2.0",
                "refiner_ref": "super-secret-value-ref",
            },
            "dependencies": [{"name": "pkg", "version": "super-secret-value-dep"}],
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl").map(
        lambda row: {"token": "super-secret-value", "x": row["x"]}
    )
    pipeline.launch_cloud(name="demo cloud", secrets={"OPENAI_API_KEY": secret})

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert "REDACTED_SECRET" in request.plan["stages"][0]["steps"][1]["args"]["fn"]
    assert secret not in request.plan["stages"][0]["steps"][1]["args"]["fn"]
    assert request.manifest is not None
    assert request.manifest["script"]["path"] == "/tmp/super-secret-value_job.py"
    assert "REDACTED_SECRET" in request.manifest["script"]["text"]
    assert secret not in request.manifest["script"]["text"]
    assert request.manifest["environment"]["refiner_version"] == "0.2.0"
    assert request.manifest["environment"]["refiner_ref"] == "super-secret-value-ref"
    assert request.manifest["dependencies"] == [
        {"name": "pkg", "version": "super-secret-value-dep"}
    ]


def test_pipeline_launch_cloud_requires_missing_env_secret(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):  # pragma: no cover
            raise AssertionError("cloud_submit_job should not be called")

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
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

    class FakeMacrodataClient:
        def __init__(self):
            raise MacrodataCredentialsError("No credentials found", missing=True)

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
    pipeline = read_jsonl("input.jsonl")

    try:
        pipeline.launch_cloud(name="demo cloud", secrets={"OPENAI_API_KEY": None})
    except SystemExit as err:
        assert "MACRODATA_API_KEY" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")


def test_pipeline_launch_cloud_requires_valid_api_key(monkeypatch) -> None:
    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_submit_job(self, *, request):  # noqa: ANN001
            del request
            raise MacrodataCredentialsError("Invalid API key", missing=False)

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    with pytest.raises(
        SystemExit,
        match="Your Macrodata API key is invalid.*with a valid key",
    ):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


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

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
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
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl")
    pipeline.launch_cloud(name="demo cloud")

    request = cast(CloudRunCreateRequest, captured["request"])
    assert [payload.stage_index for payload in request.stage_payloads] == [0, 1]
    runtimes = [payload.runtime for payload in request.stage_payloads]
    assert all(runtime is not None for runtime in runtimes)
    assert [runtime.num_workers for runtime in runtimes if runtime is not None] == [
        2,
        5,
    ]
    assert (
        request.stage_payloads[0].pipeline_payload.sha256
        != request.stage_payloads[1].pipeline_payload.sha256
    )


def test_pipeline_launch_cloud_interactive_ref_fallback_accepts(monkeypatch) -> None:
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest={
            "version": 1,
            "environment": {"refiner_version": "0.2.0", "refiner_ref": "deadbeef"},
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: False,
    )
    monkeypatch.setattr("refiner.launchers.cloud.stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    manifest = cast(dict[str, object], request.manifest)
    environment = cast(dict[str, object], manifest["environment"])
    assert environment["refiner_ref"] is None
    assert environment["refiner_version"] == "0.2.0"


def test_pipeline_launch_cloud_interactive_ref_fallback_rejects(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {
            "version": 1,
            "environment": {"refiner_version": "0.2.0", "refiner_ref": "deadbeef"},
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: False,
    )
    monkeypatch.setattr("refiner.launchers.cloud.stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    with pytest.raises(SystemExit, match="aborted"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_noninteractive_ref_fallback_env_override(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest={
            "version": 1,
            "environment": {"refiner_version": "0.2.0", "refiner_ref": "deadbeef"},
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: False,
    )
    monkeypatch.setattr("refiner.launchers.cloud.stdin_is_interactive", lambda: False)
    monkeypatch.setenv("MACRODATA_FALLBACK_TO_LATEST_PYPI", "1")

    read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    manifest = cast(dict[str, object], request.manifest)
    environment = cast(dict[str, object], manifest["environment"])
    assert environment["refiner_ref"] is None


def test_pipeline_launch_cloud_noninteractive_ref_fallback_requires_override(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda **_: {
            "version": 1,
            "environment": {"refiner_version": "0.2.0", "refiner_ref": "deadbeef"},
        },
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: False,
    )
    monkeypatch.setattr("refiner.launchers.cloud.stdin_is_interactive", lambda: False)
    monkeypatch.delenv("MACRODATA_FALLBACK_TO_LATEST_PYPI", raising=False)

    with pytest.raises(SystemExit, match="MACRODATA_FALLBACK_TO_LATEST_PYPI=1"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_detached_mode_prints_followup_commands(
    monkeypatch, capsys
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.resolve_launcher_attach_mode",
        lambda **_: "detach",
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.emit_cloud_followup_commands",
        lambda *, context, file=None: print(f"attach {context.job_id}", file=file),
    )

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")
    out = capsys.readouterr()

    assert result.job_id == "job-123"
    assert "attach job-123" in out.out


def test_pipeline_launch_cloud_attached_mode_calls_attach(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.resolve_launcher_attach_mode",
        lambda **_: "attach",
    )
    captured: dict[str, object] = {}

    def _fake_attach_to_cloud_job(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job", _fake_attach_to_cloud_job
    )

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert result.job_id == "job-123"
    assert captured["job_id"] == "job-123"
    assert captured["stage_index_hint"] == 0
    assert captured["force_attach"] is True


def test_pipeline_launch_cloud_explicit_attach_exits_nonzero_when_remote_job_fails(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setenv("REFINER_ATTACH", "attach")
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: 1,
    )

    with pytest.raises(SystemExit, match="1"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_attach_failure_prints_fallback(
    monkeypatch, capsys
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.resolve_launcher_attach_mode",
        lambda **_: "attach",
    )
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: (_ for _ in ()).throw(
            MacrodataApiError(status=503, message="boom")
        ),
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.emit_cloud_followup_commands",
        lambda *, context, file=None: print(f"fallback {context.job_id}", file=file),
    )

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    out = capsys.readouterr()
    assert result.job_id == "job-123"
    assert "attach failed" in out.err
    assert "fallback job-123" in out.err


def test_pipeline_launch_cloud_unexpected_attach_failure_propagates(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.resolve_launcher_attach_mode",
        lambda **_: "attach",
    )
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_defaults_to_auto_attach_when_interactive(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.delenv("REFINER_ATTACH", raising=False)
    monkeypatch.setattr("refiner.launchers.cloud.stdout_is_interactive", lambda: True)
    captured: dict[str, object] = {}

    def _fake_attach_to_cloud_job(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job", _fake_attach_to_cloud_job
    )

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert result.job_id == "job-123"
    assert captured["job_id"] == "job-123"
    assert captured["force_attach"] is True


def test_pipeline_launch_cloud_interactive_auto_attach_nonzero_raises(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.delenv("REFINER_ATTACH", raising=False)
    monkeypatch.setattr("refiner.launchers.cloud.stdout_is_interactive", lambda: True)
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: 1,
    )

    with pytest.raises(SystemExit, match="1"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_preserves_reducer_stage_resource_opt_out(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    stage_zero = read_jsonl("input-a.jsonl")
    stage_one = read_jsonl("input-b.jsonl")
    monkeypatch.setattr(
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=stage_zero,
                compute=StageComputeRequirements(num_workers=2),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=stage_one,
                compute=StageComputeRequirements(
                    num_workers=1,
                    inherit_launcher_resources=False,
                ),
            ),
        ],
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        cpus_per_worker=4,
        mem_mb_per_worker=8192,
        gpus_per_worker=1,
        gpu_type="h100",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.plan["stages"][0]["cpus_per_worker"] == 4
    assert request.plan["stages"][0]["memory_mb_per_worker"] == 8192
    assert request.plan["stages"][0]["gpus_per_worker"] == 1
    assert request.plan["stages"][0]["gpu_type"] == "h100"
    assert "cpus_per_worker" not in request.plan["stages"][1]
    assert "memory_mb_per_worker" not in request.plan["stages"][1]
    assert "gpus_per_worker" not in request.plan["stages"][1]
    assert "gpu_type" not in request.plan["stages"][1]
    first_runtime = request.stage_payloads[0].runtime
    second_runtime = request.stage_payloads[1].runtime
    assert first_runtime is not None
    assert second_runtime is not None
    assert first_runtime.cpus_per_worker == 4
    assert first_runtime.mem_mb_per_worker == 8192
    assert first_runtime.gpus_per_worker == 1
    assert first_runtime.gpu_type == "h100"
    assert second_runtime.cpus_per_worker is None
    assert second_runtime.mem_mb_per_worker is None
    assert second_runtime.gpus_per_worker is None
    assert second_runtime.gpu_type is None


def test_pipeline_launch_cloud_auto_attach_uses_stdout_interactivity(
    monkeypatch, capsys
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setenv("REFINER_ATTACH", "auto")
    monkeypatch.setattr("refiner.launchers.cloud.stdin_is_interactive", lambda: True)
    monkeypatch.setattr("refiner.launchers.cloud.stdout_is_interactive", lambda: False)
    monkeypatch.setattr(
        "refiner.launchers.cloud.emit_cloud_followup_commands",
        lambda *, context, file=None: print(f"attach {context.job_id}", file=file),
    )
    monkeypatch.setattr(
        "refiner.cli.run.cloud.attach_to_cloud_job",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not attach")),
    )

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")
    out = capsys.readouterr()

    assert result.job_id == "job-123"
    assert "attach job-123" in out.out


def test_pipeline_launch_cloud_continue_from_job_posts_resume_request(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, manifest={"version": 1})
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    result = read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        num_workers=3,
        cpus_per_worker=2,
        continue_from_job=" job-previous ",
    )

    assert result.job_id == "job-123"
    assert result.stage_index == 1
    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.continue_from_job == "job-previous"
    assert request.name == "demo cloud"
    assert request.manifest == {"version": 1}
    assert request.plan is not None
    assert "requested_num_workers" not in request.plan["stages"][0]
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].stage_index == 0
    assert request.stage_payloads[0].pipeline_payload.sha256 == "abc123"
    assert request.stage_payloads[0].runtime is None
    assert request.runtime_overrides is not None
    assert request.runtime_overrides.to_dict() == {
        "num_workers": 3,
        "cpus_per_worker": 2,
    }
    assert request.sync_local_dependencies is True


def test_pipeline_launch_cloud_continue_from_job_stage_posts_resume_request(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="job-previous:2",
    )

    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.continue_from_job == "job-previous:2"
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].runtime is None
    assert request.runtime_overrides is None


def test_pipeline_launch_cloud_continue_from_job_infer_posts_resume_request(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="infer",
    )

    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.continue_from_job == "infer"
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].runtime is None
    assert request.runtime_overrides is None


def test_pipeline_launch_cloud_continue_without_overrides_preserves_prior_runtime(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="job-previous",
    )

    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.runtime_overrides is None
    assert request.plan is not None
    assert "requested_num_workers" not in request.plan["stages"][0]


def test_pipeline_launch_cloud_continue_preserves_fixed_reducer_stage_runtime(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    stage_zero = read_jsonl("input-a.jsonl")
    reducer_stage = read_jsonl("input-b.jsonl")
    monkeypatch.setattr(
        "refiner.launchers.cloud.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=stage_zero,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=reducer_stage,
                compute=StageComputeRequirements(
                    num_workers=1,
                    inherit_launcher_resources=False,
                ),
            ),
        ],
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        num_workers=4,
        continue_from_job="job-previous",
    )

    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.plan is not None
    assert "requested_num_workers" not in request.plan["stages"][0]
    assert request.plan["stages"][1]["requested_num_workers"] == 1


def test_pipeline_launch_cloud_continue_rejects_invalid_stage_index(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True, fail_on_resume=True)

    with pytest.raises(
        ValueError, match="continue_from_job stage index must be an integer"
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="job-1:not-an-int",
        )


def test_pipeline_launch_cloud_continue_rejects_empty_stage_index(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True, fail_on_resume=True)

    with pytest.raises(
        ValueError, match="continue_from_job stage index must be non-empty"
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="job-1:",
        )


def test_pipeline_launch_cloud_continue_rejects_multiple_colons(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True, fail_on_resume=True)

    with pytest.raises(
        ValueError,
        match="continue_from_job must be JOBID, JOBID:stage_index, or 'infer'",
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="job-1:2:3",
        )


def test_pipeline_launch_cloud_continue_requires_selector_for_force_continue(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True, fail_on_resume=True)

    with pytest.raises(ValueError, match="force_continue requires continue_from_job"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            force_continue=True,
        )


def test_pipeline_launch_cloud_continue_forwards_force_continue(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_submit=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="job-previous",
        force_continue=True,
    )

    request = cast(CloudRunResumeRequest, captured["resume_request"])
    assert request.force_continue is True


def test_pipeline_launch_cloud_continue_rejects_blank_selector(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True, fail_on_resume=True)

    with pytest.raises(ValueError, match="continue_from_job must be non-empty"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="   ",
        )
