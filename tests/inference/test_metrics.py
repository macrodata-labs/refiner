from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, cast

import pytest

import refiner as mdr
from refiner.inference import (
    InferenceResponse,
    OpenAIEndpointProvider,
    VLLMProvider,
    generate_pooling,
)
from refiner.pipeline.data.row import DictRow
from refiner.services import VLLMRuntimeServiceBinding
from refiner.services.manager import ServiceManager
from refiner.worker.context import set_active_run_context

from ._helpers import (
    _MetricRecordingEmitter,
    openai_provider,
)


def test_inference_generate_text_reports_success_metrics(monkeypatch) -> None:
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

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(raw_payload={"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate_text(
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


def test_inference_generate_text_reports_failed_requests(monkeypatch) -> None:
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

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(raw_payload={"prompt": row["prompt"]})
        return {"output": response.text}

    infer = mdr.inference.generate_text(
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


def test_inference_request_limit_applies_before_vllm_service_ready(
    monkeypatch,
) -> None:
    emitter = _MetricRecordingEmitter()

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

    async def _fake_pooling(self, payload):
        del self, payload
        return {"data": []}

    async def _inference_fn(row, generate_pooling_request):
        await generate_pooling_request({"task": "token_classify"})
        return row

    async def _invoke() -> None:
        service_requested = asyncio.Event()
        service_ready = asyncio.Event()
        get_calls = 0
        provider = VLLMProvider(model="test-model")
        expected_service_name = provider.service_definition().name

        class _ServiceManager:
            async def get(self, name: str):
                nonlocal get_calls
                assert name == expected_service_name
                get_calls += 1
                service_requested.set()
                await service_ready.wait()
                return VLLMRuntimeServiceBinding(
                    name=name,
                    kind="llm",
                    endpoint="http://127.0.0.1:8000",
                )

        infer = generate_pooling(
            fn=_inference_fn,
            provider=provider,
            max_concurrent_requests=1,
        )

        with (
            set_active_run_context(
                job_id="job-1",
                stage_index=0,
                worker_id="worker-1",
                worker_name=None,
                runtime_lifecycle=_Runtime(),
                service_manager=cast(ServiceManager, _ServiceManager()),
                user_metrics_emitter=emitter,
            ),
        ):
            first = asyncio.create_task(
                cast(
                    Coroutine[Any, Any, object],
                    infer(DictRow({"id": 1}).with_shard_id("s")),
                )
            )
            await asyncio.wait_for(service_requested.wait(), timeout=1.0)
            second = asyncio.create_task(
                cast(
                    Coroutine[Any, Any, object],
                    infer(DictRow({"id": 2}).with_shard_id("s")),
                )
            )
            await asyncio.sleep(0)

            gauges = {
                item["label"]: item["callback"] for item in emitter.registered_gauges
            }
            assert get_calls == 1
            assert gauges["waiting_requests"]() == 1
            assert gauges["running_requests"]() == 0

            service_ready.set()
            await asyncio.gather(first, second)

    monkeypatch.setattr(openai_provider._OpenAIEndpointClient, "pooling", _fake_pooling)

    asyncio.run(_invoke())
