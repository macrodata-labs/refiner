from __future__ import annotations

import asyncio
import importlib
from typing import Any, cast

import refiner as mdr
from refiner.inference.providers import openai as openai_provider
from refiner.pipeline.data.row import DictRow
from refiner.services import VLLMRuntimeServiceBinding


def test_generate_pooling_calls_vllm_pooling_endpoint(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_pooling(self, payload):
        seen["max_connections"] = self.max_connections
        seen["max_keepalive_connections"] = self.max_keepalive_connections
        seen["payload"] = dict(payload)
        return {"data": [[1.0, 2.0, 3.0]]}

    class _FakeServiceManager:
        async def get(self, service_name: str) -> VLLMRuntimeServiceBinding:
            return VLLMRuntimeServiceBinding(
                name=service_name,
                kind="llm",
                endpoint="http://127.0.0.1:8000",
            )

    runtime_module = importlib.import_module("refiner.inference.internal.runtime")
    monkeypatch.setattr(openai_provider._OpenAIEndpointClient, "pooling", _fake_pooling)
    monkeypatch.setattr(
        runtime_module, "get_active_service_manager", lambda: _FakeServiceManager()
    )

    async def _map(row, generate_pooling):
        response = await generate_pooling(
            {
                "task": "token_classify",
                "messages": [{"role": "user", "content": row["text"]}],
            }
        )
        return {"data": response["data"]}

    infer = mdr.inference.generate_pooling(
        fn=_map,
        provider=mdr.inference.VLLMProvider(model="robometer-test"),
        max_concurrent_requests=512,
    )

    result = asyncio.run(cast(Any, infer(DictRow({"text": "open the drawer"}))))

    assert result == {"data": [[1.0, 2.0, 3.0]]}
    assert seen["payload"] == {
        "model": "robometer-test",
        "task": "token_classify",
        "messages": [{"role": "user", "content": "open the drawer"}],
    }
    assert seen["max_connections"] == 512
    assert seen["max_keepalive_connections"] == 512
