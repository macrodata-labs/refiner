from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest

import refiner as mdr
from refiner.inference import (
    AnthropicEndpointProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    anthropic_provider,
    openai_provider,
)


def test_inference_generate_text_accepts_raw_payload_and_merges_default_params(
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
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(raw_payload={"prompt": row["prompt"]})
        return {
            "output": response.text,
            "finish_reason": response.finish_reason,
        }

    infer = mdr.inference.generate_text(
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


def test_inference_generate_text_accepts_vercel_style_multimodal_messages(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="combined",
            finish_reason="stop",
            usage={"prompt_tokens": 5},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Combine these two images into one composition.",
                        },
                        {
                            "type": "file",
                            "mediaType": "image/png",
                            "data": row["first_image"],
                        },
                        {
                            "type": "file",
                            "mediaType": "image/jpeg",
                            "data": "https://example.com/second.jpg",
                        },
                    ],
                },
            ],
            temperature=0,
            providerOptions={"openai": {"reasoningEffort": "low"}},
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
        default_generation_params={"max_tokens": 256},
    )

    async def _invoke() -> object:
        return await infer(DictRow({"first_image": b"png-bytes"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "combined"}
    assert seen["payload"] == {
        "model": "gpt-test",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Combine these two images into one composition.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,cG5nLWJ5dGVz",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/second.jpg"},
                    },
                ],
            },
        ],
        "temperature": 0,
        "reasoning_effort": "low",
    }


def test_inference_generate_text_maps_openai_options_to_wire_names(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
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

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            messages=[{"role": "user", "content": "hello"}],
            providerOptions={
                "openai": {
                    "logitBias": {"42": -1},
                    "logprobs": 3,
                    "parallelToolCalls": False,
                    "maxCompletionTokens": 64,
                    "serviceTier": "flex",
                    "promptCacheKey": "cache-key",
                    "promptCacheRetention": "24h",
                    "safetyIdentifier": "safe-id",
                    "textVerbosity": "low",
                }
            },
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    asyncio.run(cast(Any, infer(DictRow({}))))

    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["logit_bias"] == {"42": -1}
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 3
    assert payload["parallel_tool_calls"] is False
    assert payload["max_completion_tokens"] == 64
    assert payload["service_tier"] == "flex"
    assert payload["prompt_cache_key"] == "cache-key"
    assert payload["prompt_cache_retention"] == "24h"
    assert payload["safety_identifier"] == "safe-id"
    assert payload["verbosity"] == "low"
    assert "providerOptions" not in payload
    assert "logitBias" not in payload


def test_inference_generate_text_accepts_text_message(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="hello",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[{"role": "user", "content": f"Summarize {row['topic']}."}]
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"topic": "logs"}))

    result = asyncio.run(_invoke())

    assert result == {"output": "hello"}
    assert seen["payload"] == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Summarize logs."}],
    }


def test_inference_generate_text_returns_provider_option_warnings(
    monkeypatch,
) -> None:
    async def _fake_generate(self, payload):
        del self, payload
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "hello"}],
            providerOptions={
                "google": {"thinkingConfig": {"thinkingBudget": 128}},
                "openai": {"strictJsonSchema": True},
            },
        )
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {
        "warnings": [
            {
                "type": "unsupported-provider-option",
                "setting": "providerOptions.google",
                "message": (
                    "'google' provider options are not used by OpenAIEndpointProvider."
                ),
            },
            {
                "type": "unsupported-setting",
                "setting": "providerOptions.openai.strictJsonSchema",
                "message": (
                    "'strictJsonSchema' is not currently mapped by "
                    "OpenAIEndpointProvider."
                ),
            },
        ]
    }


def test_inference_generate_text_warns_for_openai_model_capabilities(
    monkeypatch,
) -> None:
    async def _fake_generate(self, payload):
        del self, payload
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "hello"}],
            providerOptions={
                "openai": {
                    "reasoningEffort": "high",
                    "serviceTier": "flex",
                }
            },
        )
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-3.5-turbo",
        ),
    )

    result = asyncio.run(cast(Any, infer(DictRow({}))))

    assert result == {
        "warnings": [
            {
                "type": "unsupported-setting",
                "setting": "providerOptions.openai.reasoningEffort",
                "message": (
                    "OpenAIEndpointProvider model 'gpt-3.5-turbo' is not known "
                    "to support reasoningEffort."
                ),
                "details": (
                    "AI SDK only enables reasoning effort for known reasoning models."
                ),
            },
            {
                "type": "unsupported-setting",
                "setting": "providerOptions.openai.serviceTier",
                "message": (
                    "OpenAIEndpointProvider model 'gpt-3.5-turbo' is not known "
                    "to support flex service tier."
                ),
                "details": (
                    "AI SDK enables flex processing for o3, o4-mini, and GPT-5 "
                    "non-chat models."
                ),
            },
        ]
    }


def test_inference_generate_text_warns_for_anthropic_model_capabilities(
    monkeypatch,
) -> None:
    async def _fake_generate_text(self, payload):
        del self, payload
        return InferenceResponse(
            text="ok",
            finish_reason="end_turn",
            usage={},
            response={"content": [{"type": "text", "text": "ok"}]},
        )

    monkeypatch.setattr(
        anthropic_provider._AnthropicEndpointClient,
        "generate_text",
        _fake_generate_text,
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8192,
            providerOptions={
                "anthropic": {
                    "thinking": {"type": "adaptive"},
                    "effort": "xhigh",
                }
            },
        )
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(
            model="claude-3-haiku-20240307",
        ),
    )

    result = asyncio.run(cast(Any, infer(DictRow({}))))

    assert result == {
        "warnings": [
            {
                "type": "unsupported-setting",
                "setting": "providerOptions.anthropic.thinking",
                "message": (
                    "AnthropicEndpointProvider model 'claude-3-haiku-20240307' "
                    "is not known to support adaptive thinking."
                ),
                "details": (
                    "AI SDK only enables adaptive thinking for Claude 4.6+ "
                    "model families."
                ),
            },
            {
                "type": "unsupported-setting",
                "setting": "providerOptions.anthropic.effort",
                "message": (
                    "AnthropicEndpointProvider model 'claude-3-haiku-20240307' "
                    "is not known to support xhigh effort."
                ),
                "details": "AI SDK only enables xhigh effort for Claude Opus 4.7+.",
            },
            {
                "type": "unsupported-setting",
                "setting": "max_tokens",
                "message": (
                    "AnthropicEndpointProvider model 'claude-3-haiku-20240307' "
                    "is known to support at most 4096 output tokens."
                ),
                "details": "requested 8192",
            },
        ]
    }


def test_inference_generate_text_requires_messages_or_raw_payload() -> None:
    async def _inference_fn(row, generate_text):
        del row
        await generate_text()
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    with pytest.raises(ValueError, match="pass exactly one of messages or raw_payload"):
        asyncio.run(_invoke())


def test_inference_generate_text_rejects_raw_payload_with_typed_options() -> None:
    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            raw_payload={"messages": [{"role": "user", "content": "hello"}]},
            providerOptions={"openai": {"serviceTier": "flex"}},
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    with pytest.raises(
        ValueError, match="providerOptions are not supported with raw_payload"
    ):
        asyncio.run(_invoke())


def test_inference_generate_text_passes_custom_openai_content(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason=None,
            usage={},
            response={"output_text": "ok"},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {
                            "type": "custom",
                            "provider": "openai-responses",
                            "data": {"type": "input_image", "image_url": "file_123"},
                        },
                    ],
                }
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-5-mini"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this."},
                    {"type": "input_image", "image_url": "file_123"},
                ],
            }
        ],
    }


def test_inference_generate_text_warns_for_unsupported_image_model(
    monkeypatch,
) -> None:
    async def _fake_generate(self, payload):
        del payload
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "mediaType": "image/png",
                            "image": row["image"],
                        },
                    ],
                }
            ],
        )
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-3.5-turbo"
        ),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"image": b"\x89PNG\r\n\x1a\n"}))

    result = asyncio.run(_invoke())

    assert result == {
        "warnings": [
            {
                "type": "unsupported-content",
                "setting": "messages[0].content[1]",
                "message": (
                    "OpenAIEndpointProvider model 'gpt-3.5-turbo' is not known "
                    "to support image input."
                ),
                "details": "image/png",
            }
        ]
    }
