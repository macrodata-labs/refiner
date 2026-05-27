from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel

import refiner as mdr
from refiner.inference import (
    AnthropicEndpointProvider,
    GoogleEndpointProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
    VLLMProvider,
)
from refiner.services import VLLMRuntimeServiceBinding
from refiner.pipeline.data.row import DictRow
from refiner.worker.context import set_active_run_context
from refiner.worker.metrics.emitter import UserMetricsEmitter

from refiner.inference import client as client_module
from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider

generate_module = importlib.import_module("refiner.inference.generate")
runtime_module = importlib.import_module("refiner.inference.internal.runtime")
transport_module = importlib.import_module("refiner.inference.internal.transport")


class _Caption(BaseModel):
    title: str
    objects: list[str]


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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            prompt="hello",
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


def test_inference_generate_text_accepts_prompt(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="hello",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        response = await generate_text(prompt=f"Summarize {row['topic']}.")
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
        "prompt": "Summarize logs.",
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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            prompt="hello",
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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            prompt="hello",
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
            api_key="secret",
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
        client_module._AnthropicEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            prompt="hello",
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
            api_key="secret",
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


def test_inference_generate_text_parses_pydantic_schema_for_openai(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Desk","objects":["lamp","book"]}',
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(prompt="caption", schema=_Caption)
        assert isinstance(response.object, _Caption)
        return {
            "title": response.object.title,
            "objects": response.object.objects,
        }

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {
        "title": "Desk",
        "objects": ["lamp", "book"],
    }
    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["messages"] == [{"role": "user", "content": "caption"}]
    assert "prompt" not in payload
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "_Caption",
            "schema": _Caption.model_json_schema(),
            "strict": True,
        },
    }


def test_inference_generate_text_applies_schema_for_google(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Video","objects":["car"]}',
            finish_reason="STOP",
            usage={},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(prompt="caption", schema=_Caption)
        assert isinstance(response.object, _Caption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"title": "Video"}
    payload = cast(Mapping[str, object], seen["payload"])
    generation_config = cast(Mapping[str, object], payload["generationConfig"])
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseSchema"] == _Caption.model_json_schema()


def test_inference_generate_text_applies_schema_for_openai_responses(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Answer","objects":[]}',
            finish_reason=None,
            usage={},
            response={"output": []},
        )

    monkeypatch.setattr(
        client_module._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(prompt="caption", schema=_Caption)
        assert isinstance(response.object, _Caption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-test", api_key="secret"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"title": "Answer"}
    payload = cast(Mapping[str, object], seen["payload"])
    text = cast(Mapping[str, object], payload["text"])
    assert text["format"] == {
        "type": "json_schema",
        "name": "_Caption",
        "schema": _Caption.model_json_schema(),
        "strict": True,
    }


def test_inference_generate_text_warns_for_anthropic_schema_fallback(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Doc","objects":["chart"]}',
            finish_reason="end_turn",
            usage={},
            response={"content": []},
        )

    monkeypatch.setattr(
        client_module._AnthropicEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(prompt="caption", schema=_Caption)
        assert isinstance(response.object, _Caption)
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-test", api_key="secret"),
    )

    result = asyncio.run(cast(Any, infer(DictRow({}))))
    assert result == {
        "warnings": [
            {
                "type": "unsupported-setting",
                "setting": "schema",
                "message": (
                    "AnthropicEndpointProvider does not enforce schema natively; "
                    "Refiner adds a JSON instruction and validates the response "
                    "locally."
                ),
            }
        ]
    }
    payload = cast(Mapping[str, object], seen["payload"])
    system = cast(list[Mapping[str, object]], payload["system"])
    assert "Return only valid JSON" in cast(str, system[0]["text"])


def test_inference_generate_text_preserves_anthropic_system_with_schema(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Doc","objects":["chart"]}',
            finish_reason="end_turn",
            usage={},
            response={"content": []},
        )

    monkeypatch.setattr(
        client_module._AnthropicEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            messages=[
                {"role": "system", "content": "Use short labels."},
                {"role": "user", "content": "caption"},
            ],
            schema=_Caption,
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-test", api_key="secret"),
    )

    asyncio.run(cast(Any, infer(DictRow({}))))

    payload = cast(Mapping[str, object], seen["payload"])
    system = cast(list[Mapping[str, object]], payload["system"])
    assert system[0] == {"type": "text", "text": "Use short labels."}
    assert "Return only valid JSON" in cast(str, system[1]["text"])


def test_inference_generate_text_raises_on_schema_validation_error(
    monkeypatch,
) -> None:
    async def _fake_generate(self, payload):
        del self, payload
        return InferenceResponse(
            text='{"title":"Desk"}',
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(prompt="caption", schema=_Caption)
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    with pytest.raises(mdr.inference.InferenceSchemaValidationError):
        asyncio.run(cast(Any, infer(DictRow({}))))


def test_google_endpoint_provider_builtin_args_do_not_include_api_key() -> None:
    provider = GoogleEndpointProvider(
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="secret",
    )

    assert provider.to_builtin_args() == {
        "type": "google_endpoint",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
    }


def test_inference_generate_text_converts_messages_for_google(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="video summary",
            finish_reason="STOP",
            usage={"prompt_tokens": 9},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {"role": "system", "content": "You are a video annotator."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this video."},
                        {
                            "type": "file",
                            "mediaType": "video/mp4",
                            "data": row["video"],
                        },
                    ],
                },
            ],
            temperature=0.1,
        )
        return {"summary": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
        default_generation_params={"max_tokens": 128},
    )

    async def _invoke() -> object:
        return await infer(DictRow({"video": b"mp4-bytes"}))

    result = asyncio.run(_invoke())

    assert result == {"summary": "video summary"}
    assert seen["payload"] == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Summarize this video."},
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": "bXA0LWJ5dGVz",
                        },
                    },
                ],
            }
        ],
        "systemInstruction": {"parts": [{"text": "You are a video annotator."}]},
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 128},
    }


def test_inference_generate_text_detects_google_video_media_type(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="video summary",
            finish_reason="STOP",
            usage={},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this video."},
                        {
                            "type": "file",
                            "mediaType": "video",
                            "data": row["video"],
                        },
                    ],
                },
            ],
        )
        return {"summary": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
    )

    mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00"

    async def _invoke() -> object:
        return await infer(DictRow({"video": mp4_bytes}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Summarize this video."},
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": "AAAAGGZ0eXBpc29tAAAAAA==",
                        },
                    },
                ],
            }
        ],
    }


def test_inference_generate_text_converts_google_assistant_multimodal_history(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="STOP",
            usage={},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "I inspected the image."},
                        {"type": "text", "text": "The image has one label."},
                        {
                            "type": "file",
                            "mediaType": "image",
                            "data": row["image"],
                        },
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"image": b"\x89PNG\r\n\x1a\nfake"}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"text": "I inspected the image.", "thought": True},
                    {"text": "The image has one label."},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgpmYWtl",
                        }
                    },
                ],
            },
            {"role": "user", "parts": [{"text": "Continue."}]},
        ],
    }


def test_google_endpoint_client_posts_generate_content(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 5,
                },
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

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

    response = asyncio.run(
        client_module._GoogleEndpointClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.5-flash",
            api_key="secret",
        ).generate_text({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    )

    assert response.text == "ok"
    assert response.finish_reason == "STOP"
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert seen["headers"] == {"x-goog-api-key": "secret"}
    assert seen["path"] == "models/gemini-2.5-flash:generateContent"


def test_inference_generate_text_applies_google_provider_options(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="STOP",
            usage={},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        return {
            "output": (
                await generate_text(
                    prompt="Generate an image caption.",
                    providerOptions={
                        "google": {
                            "thinkingConfig": {"thinkingBudget": 128},
                            "responseModalities": ["TEXT"],
                            "safetySettings": [
                                {
                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                    "threshold": "BLOCK_ONLY_HIGH",
                                }
                            ],
                            "cachedContent": "cachedContents/abc",
                            "serviceTier": "flex",
                        }
                    },
                )
            ).text
        }

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "contents": [
            {"role": "user", "parts": [{"text": "Generate an image caption."}]}
        ],
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 128},
            "responseModalities": ["TEXT"],
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            }
        ],
        "cachedContent": "cachedContents/abc",
        "serviceTier": "flex",
    }


def test_inference_generate_text_passes_max_retries_as_internal_option(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="STOP",
            usage={},
            response={"candidates": []},
        )

    monkeypatch.setattr(
        client_module._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(prompt="hello", maxRetries=0)
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash", api_key="secret"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"output": "ok"}
    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["__refiner_max_retries"] == 0


def test_inference_generate_text_converts_messages_for_openai_responses(
    monkeypatch,
) -> None:
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
        client_module._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this PDF."},
                        {
                            "type": "file",
                            "mediaType": "application/pdf",
                            "data": row["pdf"],
                        },
                    ],
                }
            ],
            providerOptions={
                "openai": {
                    "reasoningEffort": "low",
                    "textVerbosity": "low",
                    "store": False,
                }
            },
            max_tokens=64,
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-5-mini", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"pdf": b"pdf-bytes"}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read this PDF."},
                    {
                        "type": "input_file",
                        "filename": "part-1.pdf",
                        "file_data": "data:application/pdf;base64,cGRmLWJ5dGVz",
                    },
                ],
            }
        ],
        "max_output_tokens": 64,
        "store": False,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    }


def test_inference_generate_text_maps_openai_responses_options_to_wire_names(
    monkeypatch,
) -> None:
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
        client_module._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            prompt="hello",
            providerOptions={
                "openai": {
                    "logprobs": True,
                    "maxToolCalls": 4,
                    "parallelToolCalls": False,
                    "previousResponseId": "resp_prev",
                    "promptCacheKey": "cache-key",
                    "promptCacheRetention": "24h",
                    "safetyIdentifier": "safe-id",
                    "serviceTier": "priority",
                }
            },
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-5-mini", api_key="secret"),
    )

    asyncio.run(cast(Any, infer(DictRow({}))))

    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["top_logprobs"] == 20
    assert payload["include"] == ["message.output_text.logprobs"]
    assert payload["max_tool_calls"] == 4
    assert payload["parallel_tool_calls"] is False
    assert payload["previous_response_id"] == "resp_prev"
    assert payload["prompt_cache_key"] == "cache-key"
    assert payload["prompt_cache_retention"] == "24h"
    assert payload["safety_identifier"] == "safe-id"
    assert payload["service_tier"] == "priority"
    assert "maxToolCalls" not in payload


def test_inference_generate_text_converts_openai_responses_assistant_reasoning(
    monkeypatch,
) -> None:
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
        client_module._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Previous reasoning."},
                        {"type": "text", "text": "Previous answer."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-5-mini", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "gpt-5-mini",
        "input": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Previous reasoning."}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Previous answer."}],
            },
            {"role": "user", "content": "Continue."},
        ],
    }


def test_inference_generate_text_detects_openai_image_media_type(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="stop",
            usage={},
            response={"choices": []},
        )

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "file",
                            "mediaType": "image/*",
                            "data": row["image"],
                        },
                    ],
                }
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    png_bytes = b"\x89PNG\r\n\x1a\nfake"

    async def _invoke() -> object:
        return await infer(DictRow({"image": png_bytes}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "gpt-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgpmYWtl"},
                    },
                ],
            }
        ],
    }


def test_inference_generate_text_converts_messages_for_anthropic(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="end_turn",
            usage={},
            response={"content": [{"type": "text", "text": "ok"}]},
        )

    monkeypatch.setattr(
        client_module._AnthropicEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        response = await generate_text(
            messages=[
                {"role": "system", "content": "Use citations."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this document."},
                        {
                            "type": "file",
                            "mediaType": "application/pdf",
                            "filename": "paper.pdf",
                            "data": row["pdf"],
                            "providerOptions": {
                                "anthropic": {
                                    "title": "Paper",
                                    "citations": {"enabled": True},
                                    "cacheControl": {"type": "ephemeral"},
                                }
                            },
                        },
                    ],
                },
            ],
            providerOptions={
                "anthropic": {
                    "thinking": {"type": "enabled", "budgetTokens": 1024},
                    "metadata": {"user_id": "user-1"},
                }
            },
            max_tokens=256,
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-sonnet-4-5", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({"pdf": b"pdf-bytes"}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document."},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "cGRmLWJ5dGVz",
                        },
                        "title": "Paper",
                        "citations": {"enabled": True},
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
        "system": [{"type": "text", "text": "Use citations."}],
        "thinking": {"type": "enabled", "budgetTokens": 1024},
        "metadata": {"user_id": "user-1"},
    }


def test_inference_generate_text_converts_anthropic_assistant_reasoning(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text="ok",
            finish_reason="end_turn",
            usage={},
            response={"content": [{"type": "text", "text": "ok"}]},
        )

    monkeypatch.setattr(
        client_module._AnthropicEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Previous reasoning."},
                        {"type": "text", "text": "Previous answer.   "},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-sonnet-4-5", api_key="secret"),
    )

    async def _invoke() -> object:
        return await infer(DictRow({}))

    asyncio.run(_invoke())

    assert seen["payload"] == {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Previous reasoning."},
                    {"type": "text", "text": "Previous answer."},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
        ],
    }


def test_inference_generate_text_requires_prompt_or_messages() -> None:
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

    with pytest.raises(ValueError, match="pass exactly one of messages or prompt"):
        asyncio.run(_invoke())


def test_parse_openai_chat_response_includes_reasoning_content() -> None:
    response = openai_provider.parse_chat_response(
        {
            "choices": [
                {
                    "message": {
                        "reasoning": "think",
                        "content": "answer",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
        use_chat=True,
    )

    assert response.text == "answer"
    assert list(response.content) == [
        {"type": "reasoning", "text": "think"},
        {"type": "text", "text": "answer"},
    ]


def test_parse_openai_responses_response_includes_reasoning_content() -> None:
    response = openai_provider.parse_responses_response(
        {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "think"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
            "usage": {},
        }
    )

    assert response.text == "answer"
    assert list(response.content) == [
        {"type": "reasoning", "text": "think"},
        {"type": "text", "text": "answer"},
    ]


def test_parse_openai_responses_response_includes_rich_parts() -> None:
    response = openai_provider.parse_responses_response(
        {
            "id": "resp_123",
            "model": "gpt-5-mini",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "Example",
                                }
                            ],
                            "logprobs": [{"token": "answer"}],
                        },
                        {
                            "type": "output_image",
                            "media_type": "image/png",
                            "b64_json": "iVBORw0KGgo=",
                        },
                    ],
                },
            ],
            "usage": {},
        }
    )

    assert response.text == "answer"
    assert response.logprobs == [[{"token": "answer"}]]
    assert response.provider_metadata == {
        "openai": {"id": "resp_123", "model": "gpt-5-mini"}
    }
    assert response.content[1]["type"] == "source"
    assert response.content[1]["url"] == "https://example.com/a"
    assert response.content[2]["type"] == "image"
    assert response.content[2]["mediaType"] == "image/png"


def test_parse_google_response_includes_reasoning_content() -> None:
    response = google_provider.parse_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "think", "thought": True},
                            {"text": "answer"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        }
    )

    assert response.text == "answer"
    assert list(response.content) == [
        {"type": "reasoning", "text": "think"},
        {"type": "text", "text": "answer"},
    ]


def test_parse_google_response_includes_sources_and_files() -> None:
    response = google_provider.parse_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "answer"},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "iVBORw0KGgo=",
                                }
                            },
                        ]
                    },
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://example.com/source",
                                    "title": "Source",
                                }
                            }
                        ]
                    },
                }
            ],
            "usageMetadata": {},
        }
    )

    assert response.text == "answer"
    assert response.content[1]["type"] == "image"
    assert response.content[1]["mediaType"] == "image/png"
    assert response.content[2]["type"] == "source"
    assert response.content[2]["url"] == "https://example.com/source"


def test_parse_anthropic_response_includes_reasoning_content() -> None:
    response = anthropic_provider.parse_response(
        {
            "content": [
                {"type": "thinking", "text": "think"},
                {"type": "text", "text": "answer"},
            ],
            "usage": {},
            "stop_reason": "end_turn",
        }
    )

    assert response.text == "answer"
    assert list(response.content) == [
        {"type": "reasoning", "text": "think"},
        {"type": "text", "text": "answer"},
    ]


def test_parse_anthropic_response_includes_citation_sources() -> None:
    response = anthropic_provider.parse_response(
        {
            "content": [
                {
                    "type": "text",
                    "text": "answer",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://example.com/source",
                            "title": "Source",
                        }
                    ],
                }
            ],
            "usage": {},
            "stop_reason": "end_turn",
        }
    )

    assert response.text == "answer"
    assert response.content[1]["type"] == "source"
    assert response.content[1]["url"] == "https://example.com/source"


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
        client_module._OpenAIResponsesClient, "generate_text", _fake_generate_text
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
        provider=OpenAIResponsesProvider(model="gpt-5-mini", api_key="secret"),
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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

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
            base_url="https://api.example.com", model="gpt-3.5-turbo", api_key="secret"
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

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

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


def test_openai_endpoint_provider_api_key_overrides_env(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def json(self) -> Mapping[str, object]:
            return {
                "choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            del base_url, timeout
            seen["headers"] = dict(headers)

        async def post(self, path, *, json):
            del path, json
            return _FakeResponse()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

    async def _inference_fn(row, generate):
        del row
        response = await generate({"prompt": "hi"})
        return {"output": response.text}

    infer = mdr.inference.generate(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com",
            model="gpt-test",
            api_key="provider-secret",
        ),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"output": "ok"}
    assert seen["headers"] == {"Authorization": "Bearer provider-secret"}


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

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)

    response = asyncio.run(
        client_module._OpenAIEndpointClient(
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


def test_openai_endpoint_retries_on_timeout(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

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
            del base_url, headers, timeout

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] < 3:
                raise httpx.TimeoutException("timed out")
            return _FakeResponse()

    async def _fake_sleep(delay: float) -> None:
        seen["sleeps"] += 1
        assert delay in (2.0, 4.0)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        client_module._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 3, "sleeps": 2}


def test_openai_endpoint_retries_on_connect_error(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

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
            del base_url, headers, timeout

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] == 1:
                raise httpx.ConnectError("connect failed")
            return _FakeResponse()

    async def _fake_sleep(delay: float) -> None:
        seen["sleeps"] += 1
        assert delay == 2.0

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        client_module._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "prompt": "hello",
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 2, "sleeps": 1}


def test_openai_endpoint_warns_on_null_chat_content(caplog) -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Thinking Process: still reasoning",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 64},
    }
    response = openai_provider.parse_chat_response(
        raw_response,
        use_chat=True,
    )

    assert response.text == ""
    assert response.finish_reason == "length"
    assert response.usage == {"completion_tokens": 64}
    assert response.raw == raw_response
    message = response.raw["choices"][0]["message"]
    assert message["reasoning"] == "Thinking Process: still reasoning"
    assert (
        "chat completion response had null message.content; returning empty text"
        in caplog.text
    )


def test_openai_endpoint_retries_on_http_503(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    error_response = httpx.Response(
        503,
        request=request,
        json={"error": {"message": "Service unavailable"}},
    )
    success_response = httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
    )

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            del base_url, headers, timeout

        async def post(self, path, *, json):
            del path, json
            seen["calls"] += 1
            if seen["calls"] == 1:
                return error_response
            return success_response

    async def _fake_sleep(delay: float) -> None:
        assert delay == 2.0
        seen["sleeps"] += 1

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        client_module._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert seen == {"calls": 2, "sleeps": 1}


def test_openai_endpoint_respects_retry_after_ms(monkeypatch) -> None:
    seen: dict[str, object] = {"calls": 0, "sleeps": []}

    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    error_response = httpx.Response(
        429,
        request=request,
        json={"error": {"message": "Rate limited"}},
        headers={"retry-after-ms": "125"},
    )
    success_response = httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
        headers={"x-request-id": "req_123"},
    )

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            del base_url, headers, timeout

        async def post(self, path, *, json):
            del path, json
            seen["calls"] = cast(int, seen["calls"]) + 1
            if seen["calls"] == 1:
                return error_response
            return success_response

    async def _fake_sleep(delay: float) -> None:
        cast(list[float], seen["sleeps"]).append(delay)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    response = asyncio.run(
        client_module._OpenAIEndpointClient(
            base_url="https://api.example.com",
        ).generate(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    assert response.text == "ok"
    assert response.headers["x-request-id"] == "req_123"
    assert seen == {"calls": 2, "sleeps": [0.125]}


def test_openai_endpoint_can_disable_retries(monkeypatch) -> None:
    seen: dict[str, int] = {"calls": 0, "sleeps": 0}

    request = httpx.Request("POST", "https://api.example.com/v1/completions")
    response = httpx.Response(
        503,
        request=request,
        json={"error": {"message": "Service unavailable"}},
    )

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            del base_url, headers, timeout

        async def post(self, path, *, json):
            del path
            assert "__refiner_max_retries" not in dict(json)
            seen["calls"] += 1
            return response

    async def _fake_sleep(delay: float) -> None:
        del delay
        seen["sleeps"] += 1

    monkeypatch.setattr(client_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(
        mdr.inference.InferenceAPICallError,
        match="generation request failed with HTTP 503: Service unavailable",
    ) as err:
        asyncio.run(
            client_module._OpenAIEndpointClient(
                base_url="https://api.example.com",
            ).generate(
                {
                    "model": "gpt-test",
                    "prompt": "hello",
                    "__refiner_max_retries": 0,
                }
            )
        )

    assert err.value.status_code == 503
    assert err.value.is_retryable is True
    assert seen == {"calls": 1, "sleeps": 0}


def test_openai_endpoint_provider_builtin_args_do_not_include_api_key() -> None:
    provider = OpenAIEndpointProvider(
        base_url="https://api.example.com",
        model="gpt-test",
        api_key="secret",
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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)
    monkeypatch.setattr(
        runtime_module, "get_active_service_manager", lambda: _FakeServiceManager()
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


def test_vllm_provider_includes_supported_service_config() -> None:
    provider = VLLMProvider(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert provider.to_builtin_args() == {
        "type": "vllm",
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "config": "correctness",
    }
    assert provider.service_definition().to_spec().config == {
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "config": "correctness",
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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

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

    monkeypatch.setattr(client_module._OpenAIEndpointClient, "generate", _fake_generate)

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
