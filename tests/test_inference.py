from __future__ import annotations

import asyncio
import inspect

import pytest

import refiner as mdr
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.steps import FnAsyncRowStep
from refiner.services import (
    InferenceResponse,
    RuntimeServiceBinding,
    ServiceRegistry,
    parse_runtime_service_bindings,
)
import refiner.services.base as services_base_module


def test_llm_service_to_spec() -> None:
    service = mdr.services.llm(
        name="llm",
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_max_context=8192,
    )

    assert service.to_spec().to_dict() == {
        "name": "llm",
        "kind": "llm",
        "config": {
            "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
            "model_max_context": 8192,
        },
    }


def test_llm_endpoint_to_spec() -> None:
    service = mdr.services.llm_endpoint(
        name="llm",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
    )

    assert service.to_spec().to_dict() == {
        "name": "llm",
        "kind": "llm_endpoint",
        "config": {
            "base_url": "https://api.openai.com",
            "api_key_env": "OPENAI_API_KEY",
        },
    }


def test_parse_runtime_service_bindings_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate service binding name"):
        parse_runtime_service_bindings(
            {
                "services": [
                    {"name": "llm", "kind": "llm", "endpoint": "http://one"},
                    {"name": "llm", "kind": "llm", "endpoint": "http://two"},
                ]
            }
        )


def test_service_registry_builds_bound_llm_client() -> None:
    service = mdr.services.llm(
        name="llm",
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    )
    registry = ServiceRegistry.from_definitions(
        definitions=[service],
        bindings=[
            RuntimeServiceBinding(
                name="llm",
                kind="llm",
                endpoint="http://127.0.0.1:9000",
                headers={"Authorization": "Bearer token"},
            )
        ],
    )

    client = registry.get("llm")
    assert isinstance(client, services_base_module.BaseGenerationService)


def test_managed_llm_requires_binding() -> None:
    service = mdr.services.llm(
        name="llm",
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    )

    with pytest.raises(ValueError, match="requires executor-provided runtime bindings"):
        service.build_client(None)


def test_inference_generate_uses_wrapper_default_row_limit() -> None:
    llm = mdr.services.llm_endpoint(
        name="llm",
        base_url="https://api.example.com",
    )
    infer = mdr.inference.generate(
        service_name="llm",
        fn=lambda row, service: {"value": row["item"], "service": bool(service)},
        max_in_flight=32,
        max_concurrent_rows=5,
    )

    pipeline = mdr.from_items([1]).map_async(infer, services=[llm])
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    assert step.max_in_flight == 5
    assert step.preserve_order is True


def test_inference_generate_defaults_row_limit_to_max_in_flight() -> None:
    llm = mdr.services.llm_endpoint(
        name="llm",
        base_url="https://api.example.com",
    )
    infer = mdr.inference.generate(
        service_name="llm",
        fn=lambda row, service: {"value": row["item"], "service": bool(service)},
        max_in_flight=24,
    )

    pipeline = mdr.from_items([1]).map_async(infer, services=[llm])
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    assert step.max_in_flight == 24


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

    monkeypatch.setattr(
        services_base_module._OpenAICompatibleGenerationService,
        "generate",
        _fake_generate,
    )

    async def _inference_fn(row, service):
        response = await service.generate({"prompt": row["prompt"]})
        return {
            "output": response.text,
            "finish_reason": response.finish_reason,
        }

    llm = mdr.services.llm_endpoint(
        name="llm",
        base_url="https://api.example.com",
    )
    infer = mdr.inference.generate(
        service_name="llm",
        fn=_inference_fn,
        default_generation_params={"temperature": 0.2},
        max_in_flight=8,
    )

    pipeline = mdr.from_items([{"prompt": "hi"}]).map_async(infer, services=[llm])
    step = pipeline.pipeline_steps[-1]
    assert isinstance(step, FnAsyncRowStep)

    async def _invoke() -> object:
        outcome = step.apply_row_async(DictRow({"prompt": "hi"}))
        assert inspect.isawaitable(outcome)
        return await outcome

    result = asyncio.run(_invoke())

    assert result == {"output": "hello", "finish_reason": "stop"}
    assert seen["payload"] == {"temperature": 0.2, "prompt": "hi"}
