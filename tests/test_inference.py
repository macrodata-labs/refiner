from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from typing import Any

import pytest

import refiner as mdr
from refiner.inference import (
    DummyRequestProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    VLLMProvider,
)
from refiner.services import VLLMRuntimeServiceBinding
from refiner.pipeline.data.row import DictRow
from refiner.worker.context import set_active_run_context
from refiner.worker.metrics.emitter import UserMetricsEmitter

from refiner.inference import client as openai_module

generate_module = importlib.import_module("refiner.inference.generate")


class _MetricRecordingEmitter(UserMetricsEmitter):
    def __init__(self) -> None:
        self.counters: list[dict[str, Any]] = []
        self.gauges: list[dict[str, Any]] = []
        self.registered_gauges: list[dict[str, Any]] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        self.gauges.append(kwargs)

    def register_user_gauge(self, **kwargs) -> None:
        self.registered_gauges.append(kwargs)

    def emit_user_histogram(self, **kwargs) -> None:
        del kwargs

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def test_openai_endpoint_requires_non_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url must be non-empty"):
        OpenAIEndpointProvider(base_url=" ", model="gpt-test")


def test_openai_endpoint_requires_non_empty_model() -> None:
    with pytest.raises(ValueError, match="model must be non-empty"):
        OpenAIEndpointProvider(base_url="https://api.example.com", model=" ")


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
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
        default_generation_params={"temperature": 0.2},
    )

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "hello", "finish_reason": "stop"}
    assert seen["payload"] == {
        "model": "gpt-test",
        "temperature": 0.2,
        "prompt": "hi",
    }


def test_openai_endpoint_includes_api_key_in_requests(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [
                    {
                        "text": "ok",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            seen["base_url"] = str(base_url)
            seen["headers"] = dict(headers)
            seen["timeout"] = timeout

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_module.httpx, "AsyncClient", _FakeAsyncClient)

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    assert seen["headers"] == {"Authorization": "Bearer secret"}


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
        def __init__(self, *, base_url, headers, timeout):
            seen["base_url"] = str(base_url)
            seen["headers"] = dict(headers)
            seen["timeout"] = timeout

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_module.httpx, "AsyncClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_module._OpenAIEndpointClient(
            base_url="https://openrouter.ai/api/v1",
        ).generate(
            {
                "model": "openai/gpt-5.2",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen["base_url"] == "https://openrouter.ai/api"
    assert seen["timeout"] == 600.0
    assert seen["path"] == "v1/chat/completions"
    assert seen["payload"] == {
        "model": "openai/gpt-5.2",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openai_endpoint_provider_builtin_args_do_not_include_api_key() -> None:
    provider = OpenAIEndpointProvider(
        base_url="https://api.example.com",
        model="gpt-test",
    )

    assert provider.to_builtin_args() == {
        "type": "openai_endpoint",
        "base_url": "https://api.example.com",
        "model": "gpt-test",
    }


def test_dummy_request_provider_builtin_args() -> None:
    provider = DummyRequestProvider(
        model="dummy-local",
        response_text="ok",
    )

    assert provider.service_definition() is None
    assert provider.to_builtin_args() == {
        "type": "dummy_request",
        "model": "dummy-local",
        "response_text": "ok",
        "host": "127.0.0.1",
        "port": 0,
    }


def test_dummy_request_provider_serves_chat_completions() -> None:
    async def _inference_fn(row, generate):
        response = await generate(
            {
                "messages": [
                    {"role": "system", "content": "Return the canned answer."},
                    {"role": "user", "content": row["prompt"]},
                ]
            }
        )
        return {"output": response.text, "finish_reason": response.finish_reason}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=DummyRequestProvider(
            model="dummy-local",
            response_text="dummy response",
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"prompt": "hi"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "dummy response", "finish_reason": "stop"}


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

    monkeypatch.setattr(openai_module._OpenAIEndpointClient, "generate", _fake_generate)
    monkeypatch.setattr(
        generate_module, "get_active_service_manager", lambda: _FakeServiceManager()
    )

    provider = VLLMProvider(model="meta-llama/Llama-3.1-8B-Instruct")

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
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


def test_vllm_provider_includes_extra_kwargs_in_service_definition() -> None:
    provider = VLLMProvider(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        model_max_context=32768,
        extra_kwargs={"limit-mm-per-prompt": "video=1"},
    )

    assert provider.to_builtin_args() == {
        "type": "vllm",
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_max_context": 32768,
        "extra_kwargs": {"limit-mm-per-prompt": "video=1"},
    }
    assert provider.service_definition().to_spec().config == {
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_max_context": 32768,
        "extra_kwargs": {"limit-mm-per-prompt": "video=1"},
    }


def test_inference_generate_reports_success_metrics(monkeypatch) -> None:
    emitter = _MetricRecordingEmitter()

    async def _fake_generate(self, payload):
        del payload
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={"prompt_tokens": 11, "completion_tokens": 7},
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

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    async def _invoke() -> object:
        with (
            set_active_run_context(
                job_id="job-1",
                stage_index=0,
                worker_id="worker-1",
                worker_name=None,
                runtime_lifecycle=_Runtime(),
                service_manager=None,
                user_metrics_emitter=emitter,
            ),
        ):
            return await infer(DictRow({"prompt": "hi"}).with_shard_id("shard-1"))

    result = asyncio.run(_invoke())

    assert result == {"output": "ok"}
    counter_totals = {
        item["label"]: sum(
            entry["value"]
            for entry in emitter.counters
            if entry["label"] == item["label"]
        )
        for item in emitter.counters
    }
    assert counter_totals["successful_requests"] == 1
    assert counter_totals["prompt_tokens"] == 11
    assert counter_totals["completion_tokens"] == 7
    assert "failed_requests" not in counter_totals
    assert {item["label"] for item in emitter.registered_gauges} >= {
        "waiting_requests",
        "running_requests",
    }


def test_inference_generate_reports_failed_requests(monkeypatch) -> None:
    emitter = _MetricRecordingEmitter()

    async def _fake_generate(self, payload):
        del self, payload
        raise RuntimeError("boom")

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

    async def _inference_fn(row, generate):
        response = await generate({"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    async def _invoke() -> object:
        with (
            set_active_run_context(
                job_id="job-1",
                stage_index=0,
                worker_id="worker-1",
                worker_name=None,
                runtime_lifecycle=_Runtime(),
                service_manager=None,
                user_metrics_emitter=emitter,
            ),
        ):
            return await infer(DictRow({"prompt": "hi"}).with_shard_id("shard-1"))

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_invoke())

    counter_totals = {
        item["label"]: sum(
            entry["value"]
            for entry in emitter.counters
            if entry["label"] == item["label"]
        )
        for item in emitter.counters
    }
    assert counter_totals["failed_requests"] == 1
    assert "successful_requests" not in counter_totals
