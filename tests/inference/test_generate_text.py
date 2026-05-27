from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

import refiner as mdr
from refiner.inference import InferenceResponse, OpenAIEndpointProvider
from refiner.inference.providers import openai as openai_provider
from refiner.pipeline.data.row import DictRow


def test_generate_text_converts_typed_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_generate(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _map(row, generate_text):
        response = await generate_text(
            messages=[{"role": "user", "content": row["text"]}],
            temperature=0,
        )
        return {"text": response.text}

    infer = mdr.inference.generate_text(
        fn=_map,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )

    result = asyncio.run(cast(Any, infer(DictRow({"text": "Hello"}))))

    assert result == {"text": "ok"}
    assert seen["payload"] == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
    }


def test_generate_text_forwards_raw_payload_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def _fake_generate(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _map(row, generate_text):
        response = await generate_text(
            raw_payload={"prompt": row["text"], "temperature": 0.5},
            temperature=0.1,
            maxRetries=0,
        )
        return {"text": response.text}

    infer = mdr.inference.generate_text(
        fn=_map,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
        default_generation_params={"temperature": 0.0, "top_p": 1},
    )

    result = asyncio.run(cast(Any, infer(DictRow({"text": "Hello"}))))

    assert result == {"text": "ok"}
    assert seen["payload"] == {
        "model": "gpt-test",
        "prompt": "Hello",
        "temperature": 0.1,
        "top_p": 1,
        "__refiner_max_retries": 0,
    }


def test_generate_text_requires_one_input_mode() -> None:
    async def _missing(row, generate_text):
        del row
        await generate_text()
        return {}

    missing = mdr.inference.generate_text(
        fn=_missing,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )

    with pytest.raises(ValueError, match="pass exactly one of messages or raw_payload"):
        asyncio.run(cast(Any, missing(DictRow({}))))

    async def _both(row, generate_text):
        del row
        await generate_text(
            messages=[{"role": "user", "content": "Hello"}],
            raw_payload={"prompt": "Hello"},
        )
        return {}

    both = mdr.inference.generate_text(
        fn=_both,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )

    with pytest.raises(ValueError, match="pass exactly one of messages or raw_payload"):
        asyncio.run(cast(Any, both(DictRow({}))))


def test_generate_text_rejects_typed_features_for_raw_payload() -> None:
    async def _map(row, generate_text):
        del row
        await generate_text(
            raw_payload={"prompt": "Hello"},
            providerOptions={"openai": {"serviceTier": "flex"}},
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_map,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
        ),
    )

    with pytest.raises(
        ValueError, match="providerOptions are not supported with raw_payload"
    ):
        asyncio.run(cast(Any, infer(DictRow({}))))
