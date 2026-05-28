from __future__ import annotations

import asyncio

import pytest

import refiner as mdr
from refiner.inference import (
    GoogleEndpointProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    VLLMProvider,
)
from refiner.pipeline.data.row import DictRow
from refiner.services import VLLMRuntimeServiceBinding

from ._helpers import (
    openai_provider,
    runtime_module,
)


def test_openai_endpoint_requires_non_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url must be non-empty"):
        OpenAIEndpointProvider(base_url=" ", model="gpt-test")


def test_openai_endpoint_requires_non_empty_model() -> None:
    with pytest.raises(ValueError, match="model must be non-empty"):
        OpenAIEndpointProvider(base_url="https://api.example.com", model=" ")


def test_google_endpoint_provider_builtin_args_are_serializable() -> None:
    provider = GoogleEndpointProvider(
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    assert provider.to_builtin_args() == {
        "type": "google_endpoint",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
    }


def test_openai_endpoint_provider_builtin_args_are_serializable() -> None:
    provider = OpenAIEndpointProvider(
        base_url="https://api.example.com",
        model="gpt-test",
    )

    assert provider.to_builtin_args() == {
        "type": "openai_endpoint",
        "base_url": "https://api.example.com",
        "model": "gpt-test",
    }


def test_vllm_provider_includes_model_in_requests(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    class _FakeServiceManager:
        async def get(self, service_name: str) -> VLLMRuntimeServiceBinding:
            return VLLMRuntimeServiceBinding(
                name=service_name,
                kind="llm",
                endpoint="http://127.0.0.1:8000",
                api_key="service-secret",
            )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )
    monkeypatch.setattr(
        runtime_module, "get_active_service_manager", lambda: _FakeServiceManager()
    )

    provider = VLLMProvider(model="meta-llama/Llama-3.1-8B-Instruct")

    async def _inference_fn(row, generate_text):
        response = await generate_text(raw_payload={"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=provider,
    )

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    assert seen["payload"] == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "hi",
    }


def test_vllm_provider_includes_supported_service_config() -> None:
    provider = VLLMProvider(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert provider.to_builtin_args() == {
        "type": "vllm",
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "config": "throughput",
    }
    assert provider.service_definition().to_spec().config == {
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "config": "throughput",
    }


def test_vllm_provider_rejects_unsupported_config() -> None:
    with pytest.raises(ValueError, match="config must be 'throughput'"):
        VLLMProvider(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            config="correctness",  # type: ignore[arg-type]
        )
