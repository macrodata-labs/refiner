from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Mapping

import pytest

import refiner as mdr
from refiner.inference import (
    InferenceResponse,
    OpenAIEndpointProvider,
    VLLMProvider,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.steps import FnAsyncRowStep
from refiner.services import VLLMRuntimeServiceBinding, VLLMServiceDefinition
from refiner.worker.context import RunHandle, set_active_run_context

openai_module = importlib.import_module("refiner.inference.client")


def test_openai_endpoint_requires_non_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url must be non-empty"):
        OpenAIEndpointProvider(base_url=" ")


def test_vllm_provider_requires_non_empty_model_name_or_path() -> None:
    with pytest.raises(ValueError, match="model_name_or_path must be non-empty"):
        VLLMProvider(model_name_or_path=" ")


def test_vllm_provider_rejects_non_positive_model_max_context() -> None:
    with pytest.raises(ValueError, match="model_max_context must be > 0 when provided"):
        VLLMProvider(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", model_max_context=0
        )


def test_vllm_provider_accepts_optional_model_max_context() -> None:
    provider = VLLMProvider(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_max_context=8192,
    )

    assert provider.model_name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
    assert provider.model_max_context == 8192


def test_vllm_provider_emits_service_definition() -> None:
    provider = VLLMProvider(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_max_context=8192,
    )

    definition = provider.service_definition()
    spec = definition.to_spec()

    assert spec.kind == "llm"
    assert definition.name.startswith("vllm-")
    assert spec.name == definition.name
    assert spec.config == {
        "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "model_max_context": 8192,
    }


def test_vllm_service_definition_emits_runtime_service_spec() -> None:
    definition = VLLMServiceDefinition(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_max_context=8192,
    )

    spec = definition.to_spec()

    assert spec.kind == "llm"
    assert spec.name == definition.name
    assert spec.config == {
        "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "model_max_context": 8192,
    }


def test_inference_generate_does_not_override_async_step_defaults() -> None:
    infer = mdr.inference.generate(
        fn=lambda row, generate: {"value": row["item"], "generate": bool(generate)},
        provider=OpenAIEndpointProvider(base_url="https://api.example.com"),
        max_concurrent_requests=32,
    )

    pipeline = mdr.from_items([1]).map_async(infer)
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    assert step.max_in_flight == 16
    assert step.preserve_order is True


def test_map_async_explicitly_controls_async_step_settings() -> None:
    infer = mdr.inference.generate(
        fn=lambda row, generate: {"value": row["item"], "generate": bool(generate)},
        provider=OpenAIEndpointProvider(base_url="https://api.example.com"),
        max_concurrent_requests=32,
    )

    pipeline = mdr.from_items([1]).map_async(
        infer,
        max_in_flight=7,
        preserve_order=False,
    )
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    assert step.max_in_flight == 7
    assert step.preserve_order is False


def test_inference_generate_rejects_non_positive_max_concurrent_requests() -> None:
    with pytest.raises(ValueError, match="max_concurrent_requests must be > 0"):
        mdr.inference.generate(
            fn=lambda row, generate: row,
            provider=OpenAIEndpointProvider(base_url="https://api.example.com"),
            max_concurrent_requests=0,
        )


def test_inference_generate_invokes_user_fn_and_merges_default_params(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="hello",
            finish_reason="stop",
            usage={"prompt_tokens": 3},
            response={"choices": []},
        )

    monkeypatch.setattr(openai_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {
            "output": response.text,
            "finish_reason": response.finish_reason,
        }

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(base_url="https://api.example.com"),
        default_generation_params={"temperature": 0.2},
        max_concurrent_requests=8,
    )

    pipeline = mdr.from_items([{"prompt": "hi"}]).map_async(infer)
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    async def _invoke() -> object:
        outcome = step.apply_row_async(DictRow({"prompt": "hi"}))
        assert inspect.isawaitable(outcome)
        return await outcome

    result = asyncio.run(_invoke())

    assert result == {"output": "hello", "finish_reason": "stop"}
    assert seen["payload"] == {"temperature": 0.2, "prompt": "hi"}


def test_openai_endpoint_includes_api_key_in_requests(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        seen["api_key"] = self.api_key
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(openai_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            api_key="secret",
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    assert seen["api_key"] == "secret"


def test_openai_endpoint_preserves_base_url_path_prefix(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout, headers):
            seen["base_url"] = str(base_url)
            seen["timeout"] = timeout
            seen["headers"] = dict(headers)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_module.httpx, "AsyncClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_module._OpenAIEndpointClient(
            base_url="https://openrouter.ai/api/v1",
            api_key="secret",
        ).generate(
            {
                "model": "openai/gpt-5.2",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen["base_url"] == "https://openrouter.ai/api"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["payload"] == {
        "model": "openai/gpt-5.2",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_vllm_provider_resolves_runtime_service_binding(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        seen["base_url"] = self.base_url
        seen["headers"] = dict(self.headers or {})
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    class _Runtime:
        def claim(self, previous=None):
            del previous
            return None

        def heartbeat(self, shards):
            del shards

        def complete(self, shard):
            del shard

        def fail(self, shard, error=None):
            del shard, error

        def finalized_workers(self, *, stage_index=None):
            del stage_index
            return []

    monkeypatch.setattr(openai_module._OpenAIEndpointClient, "generate", _fake_generate)

    provider = VLLMProvider(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_max_context=8192,
    )
    binding = VLLMRuntimeServiceBinding(
        name=provider.service_definition().name,
        kind="llm",
        endpoint="http://127.0.0.1:8000",
    )

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=provider,
    )

    async def _invoke() -> object:
        with set_active_run_context(
            run_handle=RunHandle(job_id="job-1", stage_index=0, worker_id="worker-1"),
            runtime_lifecycle=_Runtime(),
            service_bindings={binding.name: binding},
        ):
            return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    assert seen["base_url"] == "http://127.0.0.1:8000"
    assert seen["headers"] == {}
