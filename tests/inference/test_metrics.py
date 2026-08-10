from __future__ import annotations

import asyncio

import pytest

import refiner as mdr
from refiner.inference import (
    InferenceResponse,
    OpenAIEndpointProvider,
)
from refiner.pipeline.data.row import DictRow
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
