from __future__ import annotations

import hashlib
import pytest
from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

import refiner as mdr
from refiner.pipeline import read_jsonl
from refiner.pipeline.resources import GPU
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements
from refiner.launchers.cloud import CloudLauncher
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudFileCompleteRequestItem,
    CloudFileUploadInstruction,
    CloudFileUploadRequestItem,
    CloudFileUploadStatus,
    CloudRunCreateRequest,
    MacrodataApiError,
    MacrodataClient,
)
from refiner.platform.client.serialize import PreparedPipelinePayload
from refiner.platform.manifest import _redact_captured_text
from refiner.launchers.secrets import Secrets

_TEST_TIMESTAMP = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)


async def _noop_inference(row, generate):
    del generate
    return row


def _prepared_payload(payload_bytes: bytes) -> PreparedPipelinePayload:
    return PreparedPipelinePayload(
        payload_bytes=payload_bytes,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        size_bytes=len(payload_bytes),
    )


def _cloud_file_upload_instruction(
    file: CloudFileUploadRequestItem,
    *,
    file_id: str = "00000000-0000-7000-8000-000000000001",
    status: CloudFileUploadStatus = CloudFileUploadStatus.NEW,
) -> CloudFileUploadInstruction:
    if status is CloudFileUploadStatus.EXISTS:
        return CloudFileUploadInstruction(
            file_id=file_id,
            sha256=file.sha256,
            size_bytes=file.size_bytes,
            status=status,
            expires_at=None,
        )
    return CloudFileUploadInstruction(
        file_id=file_id,
        sha256=file.sha256,
        size_bytes=file.size_bytes,
        status=status,
        url=f"https://payloads.example/{file_id}",
        required_headers={
            "content-length": str(file.size_bytes),
            "x-amz-checksum-sha256": "checksum",
        },
        upload_url_expires_at=_TEST_TIMESTAMP,
        expires_at=None,
    )


def _stub_cloud_submit(
    monkeypatch,
    *,
    manifest: dict[str, object] | Callable[..., dict[str, object]] | None = None,
    fail_on_submit: bool = False,
    fail_on_upload_urls: bool = False,
    fail_on_upload: bool = False,
    fail_on_complete: bool = False,
) -> dict[str, object]:
    captured: dict[str, object] = {
        "events": [],
        "uploads": [],
        "upload_url_batches": [],
        "complete_batches": [],
    }

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            if fail_on_upload_urls:
                raise MacrodataApiError(
                    status=400, message="failed to create upload URLs"
                )
            assert object_ttl_secs is None
            cast(list[str], captured["events"]).append("upload-urls")
            captured["upload_url_files"] = files
            cast(
                list[list[CloudFileUploadRequestItem]], captured["upload_url_batches"]
            ).append(files)
            offset = sum(
                len(batch)
                for batch in cast(
                    list[list[CloudFileUploadRequestItem]],
                    captured["upload_url_batches"],
                )[:-1]
            )
            instructions = []
            for index, file in enumerate(files, start=1):
                instructions.append(
                    _cloud_file_upload_instruction(
                        file,
                        file_id=f"00000000-0000-7000-8000-{offset + index:012d}",
                    )
                )

            class _Resp:
                files = instructions

            return _Resp()

        def cloud_upload_file(self, *, instruction, payload_bytes):
            if fail_on_upload:
                raise MacrodataApiError(status=502, message="failed to upload payload")
            cast(list[str], captured["events"]).append("upload")
            cast(list[object], captured["uploads"]).append((instruction, payload_bytes))

        def cloud_complete_files(self, *, files, object_ttl_secs=None):
            if fail_on_complete:
                raise MacrodataApiError(
                    status=502, message="failed to complete payload"
                )
            assert object_ttl_secs is None
            cast(list[str], captured["events"]).append("complete")
            captured["complete_files"] = files
            cast(
                list[list[CloudFileCompleteRequestItem]], captured["complete_batches"]
            ).append(files)

            class _Resp:
                files = []

            return _Resp()

        def cloud_submit_job(self, *, request):
            if fail_on_submit:
                raise AssertionError("should not submit")
            cast(list[str], captured["events"]).append("submit")
            captured["submit_request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 1 if request.continue_from_job is not None else 0
                status = "queued"
                workspace_slug = None
                warnings: list[str] = []

            return _Resp()

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(b"AQID"),
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
        gpu=GPU(count=2, type="h100", cuda_version="12.8"),
    )

    assert result.job_id == "job-123"
    assert result.stage_index == 0
    assert result.status == "queued"
    assert result.warnings == []

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.name == "demo cloud"
    assert request.sync_local_dependencies is True
    assert request.plan["stages"][0]["name"] == "stage_0"
    assert request.plan["stages"][0]["requested_num_workers"] == 3
    assert request.plan["stages"][0]["cpus_per_worker"] == 2
    assert request.plan["stages"][0]["memory_mb_per_worker"] == 4096
    assert request.plan["stages"][0]["gpu"] == {
        "count": 2,
        "type": "h100",
        "cuda_version": "12.8",
    }
    assert "cuda_version" not in request.plan["stages"][0]
    assert len(request.stage_payloads) == 1
    assert request.stage_payloads[0].stage_index == 0
    assert request.stage_payloads[0].pipeline_payload.file_id == (
        "00000000-0000-7000-8000-000000000001"
    )
    assert request.stage_payloads[0].runtime is not None
    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
    assert request.stage_payloads[0].runtime.mem_mb_per_worker == 4096
    assert request.stage_payloads[0].runtime.gpu == GPU(
        count=2,
        type="h100",
        cuda_version="12.8",
    )
    assert request.manifest == {
        "version": 1,
        "environment": {"refiner_version": "0.2.0", "refiner_ref": "abc123def456"},
        "script": {"text": "print('hi')"},
    }
    upload_url_files = cast(
        list[CloudFileUploadRequestItem], captured["upload_url_files"]
    )
    assert len(upload_url_files) == 1
    assert upload_url_files[0].sha256 == _prepared_payload(b"AQID").sha256
    uploaded_payloads = cast(
        list[tuple[CloudFileUploadInstruction, bytes]], captured["uploads"]
    )
    assert uploaded_payloads[0][1] == b"AQID"
    complete_files = cast(
        list[CloudFileCompleteRequestItem], captured["complete_files"]
    )
    assert complete_files[0].file_id == "00000000-0000-7000-8000-000000000001"
    assert captured["events"] == ["upload-urls", "upload", "complete", "submit"]


def test_pipeline_launch_cloud_embeds_runtime_services(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    pipeline = read_jsonl("input.jsonl").map_async(
        mdr.inference.generate_text(
            fn=_noop_inference,
            provider=mdr.inference.VLLMProvider(model="Qwen/Qwen3.5-9B"),
        )
    )
    pipeline.launch_cloud(name="demo cloud")

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    expected_service = {
        "name": request.plan["stages"][0]["runtime_services"][0]["name"],
        "kind": "llm",
        "config": {
            "model_name_or_path": "Qwen/Qwen3.5-9B",
            "config": "correctness",
        },
    }
    assert request.plan["stages"][0]["runtime_services"] == [expected_service]
    assert request.stage_payloads[0].runtime_services[0].to_dict() == expected_service


def test_pipeline_launch_cloud_accepts_structured_gpu(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        gpu=GPU(count=1, type="h100", cuda_version="12.4"),
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.plan["stages"][0]["gpu"] == {
        "count": 1,
        "type": "h100",
        "cuda_version": "12.4",
    }
    runtime = request.stage_payloads[0].runtime
    assert runtime.gpu == GPU(count=1, type="h100", cuda_version="12.4")


def test_gpu_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="gpu.count must be > 0"):
        GPU(count=0, type="h100")
    with pytest.raises(ValueError, match="gpu.type must be one of: h100"):
        GPU(count=1, type="a100")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gpu.cuda_version must be one of"):
        GPU(count=1, type="h100", cuda_version="13.0")  # type: ignore[arg-type]


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
    assert request.secrets == [
        {
            "OPENAI_API_KEY": "env-secret",
            "MODEL_NAME": "gpt-5",
        }
    ]


def test_pipeline_launch_cloud_accepts_env_secret_source(monkeypatch) -> None:
    captured = _stub_cloud_submit(
        monkeypatch,
        manifest={"version": 1, "script": {"text": "HF_TOKEN should not be redacted"}},
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        secrets=Secrets.env(name="production", keys=["HF_TOKEN"]),
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == [
        {"__type__": "__envkeys__", "envname": "production", "keys": ["HF_TOKEN"]}
    ]
    assert request.manifest is not None
    assert request.manifest["script"]["text"] == "HF_TOKEN should not be redacted"


def test_pipeline_launch_cloud_rejects_string_env_secret_keys() -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        Secrets.env(keys="HF_TOKEN")


def test_pipeline_launch_cloud_accepts_multiple_secret_sources(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        secrets=[
            Secrets.env(name="default"),
            Secrets.dict({"MODEL_NAME": "gpt-5"}),
        ],
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == [
        {"__type__": "__envkeys__", "envname": "default"},
        {"MODEL_NAME": "gpt-5"},
    ]


def test_pipeline_launch_cloud_loads_dotenv_as_dict_secret_source(
    monkeypatch, tmp_path
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "HF_TOKEN=hf_secret\nexport MODEL_NAME='gpt-5' # inline comment\nEMPTY=\n",
        encoding="utf-8",
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        secrets=Secrets.dotenv(dotenv_path),
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == [
        {"HF_TOKEN": "hf_secret", "MODEL_NAME": "gpt-5", "EMPTY": ""}
    ]


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
    assert request.secrets == [
        {
            "OPENAI_API_KEY": "super-secret-value",
        }
    ]
    assert request.env == {"MODEL_NAME": "plain-env-value"}
    assert "REDACTED_SECRET" in request.plan["stages"][0]["steps"][1]["args"]["fn"]


def test_pipeline_launch_cloud_allows_env_to_override_secret_keys(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        secrets={"API_KEY": "secret"},
        env={"API_KEY": "env"},
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == [{"API_KEY": "secret"}]
    assert request.env == {"API_KEY": "env"}


def test_pipeline_launch_cloud_allows_env_to_override_env_secret_keys(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        secrets=Secrets.env(keys=["API_KEY"]),
        env={"API_KEY": "env"},
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.secrets == [
        {"__type__": "__envkeys__", "envname": "default", "keys": ["API_KEY"]}
    ]
    assert request.env == {"API_KEY": "env"}


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
    except ValueError as err:
        assert "OPENAI_API_KEY" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


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

        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):  # noqa: ANN001
            del files, object_ttl_secs
            raise MacrodataCredentialsError("Invalid API key", missing=False)

        def cloud_submit_job(self, *, request):  # noqa: ANN001
            del request
            raise AssertionError("cloud_submit_job should not be called")

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


def test_pipeline_launch_cloud_upload_url_failure_prevents_submit(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_upload_urls=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    with pytest.raises(SystemExit, match="failed to create upload URLs"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert captured["uploads"] == []
    assert "submit_request" not in captured


def test_pipeline_launch_cloud_payload_upload_failure_prevents_submit(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_upload=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    with pytest.raises(SystemExit, match="failed to upload payload"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert "upload_url_files" in captured
    assert "complete_files" not in captured
    assert "submit_request" not in captured


def test_pipeline_launch_cloud_complete_failure_prevents_submit(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch, fail_on_complete=True)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    with pytest.raises(SystemExit, match="failed to complete payload"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert captured["uploads"]
    assert "submit_request" not in captured


def test_pipeline_launch_cloud_missing_upload_url_response_prevents_submit(
    monkeypatch,
) -> None:
    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            del files, object_ttl_secs

            class _Resp:
                files = []

            return _Resp()

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", FakeMacrodataClient)
    monkeypatch.setattr(
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(b"payload"),
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    with pytest.raises(SystemExit, match="did not return instructions"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")


def test_pipeline_launch_cloud_existing_cloud_file_skips_upload_and_complete(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"uploads": []}

    class FakeMacrodataClient:
        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            assert object_ttl_secs is None
            captured["upload_url_files"] = files
            instructions = [
                _cloud_file_upload_instruction(
                    files[0],
                    file_id="00000000-0000-7000-8000-000000000123",
                    status=CloudFileUploadStatus.EXISTS,
                )
            ]

            class _Resp:
                files = instructions

            return _Resp()

        def cloud_upload_file(self, *, instruction, payload_bytes):
            del instruction, payload_bytes
            raise AssertionError("existing cloud files should not be uploaded")

        def cloud_complete_files(self, *, files, object_ttl_secs=None):
            del files, object_ttl_secs
            raise AssertionError("existing cloud files should not be completed")

    monkeypatch.setattr(
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(b"payload"),
    )

    payloads = CloudLauncher._upload_stage_payloads(
        client=cast(MacrodataClient, FakeMacrodataClient()),
        stages=[
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=read_jsonl("input.jsonl"),
                compute=StageComputeRequirements(num_workers=1),
            )
        ],
    )

    assert (
        len(cast(list[CloudFileUploadRequestItem], captured["upload_url_files"])) == 1
    )
    assert captured["uploads"] == []
    assert payloads[0].file_id == "00000000-0000-7000-8000-000000000123"


def test_pipeline_launch_cloud_deduplicates_identical_stage_payloads(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"uploads": []}

    class FakeMacrodataClient:
        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            assert object_ttl_secs is None
            captured["upload_url_files"] = files
            instructions = [
                _cloud_file_upload_instruction(
                    files[0],
                    file_id="00000000-0000-7000-8000-000000000123",
                )
            ]

            class _Resp:
                files = instructions

            return _Resp()

        def cloud_upload_file(self, *, instruction, payload_bytes):
            cast(list[object], captured["uploads"]).append((instruction, payload_bytes))

        def cloud_complete_files(self, *, files, object_ttl_secs=None):
            assert object_ttl_secs is None
            captured["complete_files"] = files

            class _Resp:
                files = []

            return _Resp()

    monkeypatch.setattr(
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(b"same-payload"),
    )

    payloads = CloudLauncher._upload_stage_payloads(
        client=cast(MacrodataClient, FakeMacrodataClient()),
        stages=[
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=read_jsonl("input-a.jsonl"),
                compute=StageComputeRequirements(num_workers=1),
            ),
            PlannedStage(
                index=1,
                name="stage_1",
                pipeline=read_jsonl("input-b.jsonl"),
                compute=StageComputeRequirements(num_workers=1),
            ),
        ],
    )

    assert (
        len(cast(list[CloudFileUploadRequestItem], captured["upload_url_files"])) == 1
    )
    assert len(cast(list[object], captured["uploads"])) == 1
    assert (
        len(cast(list[CloudFileCompleteRequestItem], captured["complete_files"])) == 1
    )
    assert payloads[0].file_id == "00000000-0000-7000-8000-000000000123"
    assert payloads[1].file_id == "00000000-0000-7000-8000-000000000123"


def test_pipeline_launch_cloud_batches_more_than_100_unique_payload_files(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(f"payload-{id(pipeline)}".encode()),
    )
    stage_pipelines = [read_jsonl(f"input-{index}.jsonl") for index in range(101)]
    monkeypatch.setattr(
        "refiner.launchers.base.plan_pipeline_stages",
        lambda pipeline, default_num_workers: [
            PlannedStage(
                index=index,
                name=f"stage_{index}",
                pipeline=stage_pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            )
            for index, stage_pipeline in enumerate(stage_pipelines)
        ],
    )
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    upload_url_batches = cast(
        list[list[CloudFileUploadRequestItem]], captured["upload_url_batches"]
    )
    complete_batches = cast(
        list[list[CloudFileCompleteRequestItem]], captured["complete_batches"]
    )
    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert [len(batch) for batch in upload_url_batches] == [100, 1]
    assert [len(batch) for batch in complete_batches] == [100, 1]
    assert len(cast(list[object], captured["uploads"])) == 101
    assert len(request.stage_payloads) == 101
    assert request.stage_payloads[0].pipeline_payload.file_id == (
        "00000000-0000-7000-8000-000000000001"
    )
    assert request.stage_payloads[-1].pipeline_payload.file_id == (
        "00000000-0000-7000-8000-000000000101"
    )


def test_pipeline_launch_cloud_submits_one_stage_payload_per_planned_stage(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"uploads": []}

    class FakeMacrodataClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            assert object_ttl_secs is None
            captured["upload_url_files"] = files
            instructions = []
            for index, file in enumerate(files, start=1):
                instructions.append(
                    _cloud_file_upload_instruction(
                        file,
                        file_id=f"00000000-0000-7000-8000-{index:012d}",
                    )
                )

            class _Resp:
                files = instructions

            return _Resp()

        def cloud_upload_file(self, *, instruction, payload_bytes):
            cast(list[object], captured["uploads"]).append((instruction, payload_bytes))

        def cloud_complete_files(self, *, files, object_ttl_secs=None):
            assert object_ttl_secs is None
            captured["complete_files"] = files

            class _Resp:
                files = []

            return _Resp()

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
        "refiner.launchers.cloud.PreparedPipelinePayload.from_pipeline",
        lambda pipeline: _prepared_payload(f"payload-{id(pipeline)}".encode()),
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
        request.stage_payloads[0].pipeline_payload.file_id
        != request.stage_payloads[1].pipeline_payload.file_id
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


def test_pipeline_launch_cloud_explicit_attach_nonzero_exits(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.setenv("REFINER_ATTACH", "attach")
    captured: dict[str, object] = {}
    monkeypatch.setattr("refiner.cli.run.cloud.attach_to_cloud_job", lambda **_: 1)
    monkeypatch.setattr(
        "refiner.launchers.cloud.emit_cloud_followup_commands",
        lambda *, context, file=None: captured.update(
            {"job_id": context.job_id, "file": file}
        ),
    )

    with pytest.raises(SystemExit) as err:
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert err.value.code == 1
    assert captured == {}


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


def test_pipeline_launch_cloud_interactive_auto_attach_nonzero_returns_job(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    monkeypatch.delenv("REFINER_ATTACH", raising=False)
    monkeypatch.setattr("refiner.launchers.cloud.stdout_is_interactive", lambda: True)
    monkeypatch.setattr("refiner.cli.run.cloud.attach_to_cloud_job", lambda **_: 1)

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert result.job_id == "job-123"


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
        gpu=GPU(count=1, type="h100", cuda_version="12.6"),
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.plan["stages"][0]["cpus_per_worker"] == 4
    assert request.plan["stages"][0]["memory_mb_per_worker"] == 8192
    assert request.plan["stages"][0]["gpu"] == {
        "count": 1,
        "type": "h100",
        "cuda_version": "12.6",
    }
    assert "cpus_per_worker" not in request.plan["stages"][1]
    assert "memory_mb_per_worker" not in request.plan["stages"][1]
    assert "gpu" not in request.plan["stages"][1]
    first_runtime = request.stage_payloads[0].runtime
    second_runtime = request.stage_payloads[1].runtime
    assert first_runtime is not None
    assert second_runtime is not None
    assert first_runtime.cpus_per_worker == 4
    assert first_runtime.mem_mb_per_worker == 8192
    assert first_runtime.gpu == GPU(count=1, type="h100", cuda_version="12.6")
    assert second_runtime.cpus_per_worker is None
    assert second_runtime.mem_mb_per_worker is None
    assert second_runtime.gpu is None


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


def test_pipeline_launch_cloud_continue_from_job_posts_submit_request(
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
        continue_from_job=" 00000000-0000-1000-8000-000000000123 ",
    )

    assert result.job_id == "job-123"
    assert result.stage_index == 1
    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.continue_from_job == "00000000-0000-1000-8000-000000000123"
    assert request.name == "demo cloud"
    assert request.manifest == {"version": 1}
    assert request.plan is not None
    assert request.plan["stages"][0]["requested_num_workers"] == 3
    assert request.plan["stages"][0]["cpus_per_worker"] == 2
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].stage_index == 0
    assert request.stage_payloads[0].pipeline_payload.file_id == (
        "00000000-0000-7000-8000-000000000001"
    )
    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
    assert request.sync_local_dependencies is True


def test_pipeline_launch_cloud_continue_from_job_stage_posts_submit_request(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="00000000-0000-1000-8000-000000000123:2",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.continue_from_job == "00000000-0000-1000-8000-000000000123:2"
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].runtime.num_workers == 1


def test_pipeline_launch_cloud_continue_from_job_infer_posts_submit_request(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="infer",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.continue_from_job == "infer"
    assert request.stage_payloads is not None
    assert request.stage_payloads[0].runtime.num_workers == 1


def test_pipeline_launch_cloud_continue_uses_current_plan_runtime(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="00000000-0000-1000-8000-000000000123",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.plan is not None
    assert request.plan["stages"][0]["requested_num_workers"] == 1


def test_pipeline_launch_cloud_continue_preserves_fixed_reducer_stage_runtime(
    monkeypatch,
) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )
    stage_zero = read_jsonl("input-a.jsonl")
    reducer_stage = read_jsonl("input-b.jsonl")
    monkeypatch.setattr(
        "refiner.launchers.cloud.CloudLauncher._planned_stages",
        lambda self: [
            PlannedStage(
                index=0,
                name="stage_0",
                pipeline=stage_zero,
                compute=StageComputeRequirements(num_workers=self.num_workers),
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
        continue_from_job="00000000-0000-1000-8000-000000000123",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.plan is not None
    assert request.plan["stages"][0]["requested_num_workers"] == 4
    assert request.plan["stages"][1]["requested_num_workers"] == 1


def test_pipeline_launch_cloud_continue_rejects_invalid_stage_index(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError, match="continue_from_job stage index must be an integer"
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="00000000-0000-1000-8000-000000000123:not-an-int",
        )


def test_pipeline_launch_cloud_continue_rejects_empty_stage_index(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError, match="continue_from_job stage index must be non-empty"
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="00000000-0000-1000-8000-000000000123:",
        )


def test_pipeline_launch_cloud_continue_rejects_multiple_colons(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError,
        match="continue_from_job must be UUID, UUID:stage_index, or 'infer'",
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="00000000-0000-1000-8000-000000000123:2:3",
        )


def test_pipeline_launch_cloud_continue_requires_selector_for_unsafe_continue(
    monkeypatch,
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="unsafe_continue requires continue_from_job"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            unsafe_continue=True,
        )


def test_pipeline_launch_cloud_continue_forwards_unsafe_continue(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    read_jsonl("input.jsonl").launch_cloud(
        name="demo cloud",
        continue_from_job="00000000-0000-1000-8000-000000000123",
        unsafe_continue=True,
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    assert request.unsafe_continue is True


def test_pipeline_launch_cloud_surfaces_submit_warnings(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)
    monkeypatch.setattr(
        "refiner.launchers.cloud.refiner_ref_exists_on_remote",
        lambda ref: True,
    )

    class WarningClient:
        def __init__(self):
            self.base_url = "https://example.com"

        def cloud_create_file_upload_urls(self, *, files, object_ttl_secs=None):
            del object_ttl_secs
            instructions = []
            for index, file in enumerate(files, start=1):
                instructions.append(
                    _cloud_file_upload_instruction(
                        file,
                        file_id=f"00000000-0000-7000-8000-{index:012d}",
                    )
                )

            class _Resp:
                files = instructions

            return _Resp()

        def cloud_upload_file(self, *, instruction, payload_bytes):
            del instruction, payload_bytes

        def cloud_complete_files(self, *, files, object_ttl_secs=None):
            del files, object_ttl_secs

            class _Resp:
                files = []

            return _Resp()

        def cloud_submit_job(self, *, request):
            captured["submit_request"] = request

            class _Resp:
                job_id = "job-123"
                stage_index = 0
                status = "queued"
                workspace_slug = None
                warnings = ["Current executor settings differ"]

            return _Resp()

    monkeypatch.setattr("refiner.launchers.cloud.MacrodataClient", WarningClient)

    result = read_jsonl("input.jsonl").launch_cloud(name="demo cloud")

    assert result.warnings == ["Current executor settings differ"]


def test_pipeline_launch_cloud_continue_rejects_blank_selector(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="continue_from_job must be non-empty"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="   ",
        )


def test_pipeline_launch_cloud_continue_rejects_non_uuid_selector(monkeypatch) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(
        ValueError, match="continue_from_job must be UUID, UUID:stage_index, or 'infer'"
    ):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud",
            continue_from_job="job-previous",
        )
