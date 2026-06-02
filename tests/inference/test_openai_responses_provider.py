from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast


import refiner as mdr
from refiner.inference import (
    GoogleEndpointProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    google_provider,
    openai_provider,
)


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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "hello"}], maxRetries=0
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
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
        openai_provider._OpenAIResponsesClient, "generate_text", _fake_generate_text
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
        provider=OpenAIResponsesProvider(model="gpt-5-mini"),
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
        openai_provider._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            messages=[{"role": "user", "content": "hello"}],
            providerOptions={
                "openai": {
                    "logprobs": True,
                    "maxToolCalls": 4,
                    "parallelToolCalls": False,
                    "previousResponseId": "resp_prev",
                    "maxCompletionTokens": 128,
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
        provider=OpenAIResponsesProvider(model="gpt-5-mini"),
    )

    asyncio.run(cast(Any, infer(DictRow({}))))

    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["top_logprobs"] == 20
    assert payload["include"] == ["message.output_text.logprobs"]
    assert payload["max_tool_calls"] == 4
    assert payload["parallel_tool_calls"] is False
    assert payload["previous_response_id"] == "resp_prev"
    assert payload["max_output_tokens"] == 128
    assert payload["prompt_cache_key"] == "cache-key"
    assert payload["prompt_cache_retention"] == "24h"
    assert payload["safety_identifier"] == "safe-id"
    assert payload["service_tier"] == "priority"
    assert "maxToolCalls" not in payload


def test_parse_openai_responses_image_generation_call_without_text() -> None:
    response = openai_provider.parse_responses_response(
        {
            "output": [
                {
                    "type": "image_generation_call",
                    "id": "ig_123",
                    "status": "completed",
                    "result": "iVBORw0KGgo=",
                    "output_format": "png",
                }
            ],
            "usage": {},
        }
    )

    assert response.text == ""
    assert response.content == [
        {
            "type": "image",
            "mediaType": "image/png",
            "data": "iVBORw0KGgo=",
            "providerMetadata": {
                "openai": {
                    "id": "ig_123",
                    "type": "image_generation_call",
                    "status": "completed",
                }
            },
        }
    ]


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
        openai_provider._OpenAIResponsesClient, "generate_text", _fake_generate_text
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
        provider=OpenAIResponsesProvider(model="gpt-5-mini"),
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


def test_openai_chat_reasoning_models_strip_unsupported_settings(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }

    class _FakeAsyncClient:
        def __init__(
            self, *, base_url, headers, max_connections, max_keepalive_connections
        ):
            del base_url, headers, max_connections, max_keepalive_connections

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_provider, "_AiohttpAPIClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_provider._OpenAIEndpointClient(
            base_url="https://api.openai.com",
        ).generate(
            {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 128,
                "temperature": 0.2,
                "top_p": 0.9,
                "logprobs": True,
                "top_logprobs": 3,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "logit_bias": {"1": 2},
            }
        )
    )

    assert response.text == "ok"
    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["max_completion_tokens"] == 128
    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "logprobs",
        "top_logprobs",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
    ):
        assert key not in payload


def test_openai_gpt_51_reasoning_none_keeps_compatible_settings(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {"output_text": "ok", "usage": {}}

    class _FakeAsyncClient:
        def __init__(
            self, *, base_url, headers, max_connections, max_keepalive_connections
        ):
            del base_url, headers, max_connections, max_keepalive_connections

        async def post(self, path, *, json):
            seen["path"] = path
            seen["payload"] = dict(json)
            return _FakeResponse()

    monkeypatch.setattr(openai_provider, "_AiohttpAPIClient", _FakeAsyncClient)

    response = asyncio.run(
        openai_provider._OpenAIResponsesClient(
            base_url="https://api.openai.com",
        ).generate_text(
            {
                "model": "gpt-5.1",
                "input": "hello",
                "reasoning": {"effort": "none"},
                "temperature": 0.2,
                "top_p": 0.9,
                "logprobs": True,
                "top_logprobs": 3,
                "frequency_penalty": 0.1,
            }
        )
    )

    assert response.text == "ok"
    payload = cast(Mapping[str, object], seen["payload"])
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["logprobs"] is True
    assert "top_logprobs" not in payload
    assert "frequency_penalty" not in payload


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
        {
            "type": "reasoning",
            "text": "think",
            "providerMetadata": {
                "openai": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "think"}],
                }
            },
        },
        {
            "type": "text",
            "text": "answer",
            "providerMetadata": {"openai": {"type": "output_text"}},
        },
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
        "openai": {"id": "resp_123", "model": "gpt-5-mini", "usage": {}}
    }
    assert response.content[1]["type"] == "source"
    assert response.content[1]["url"] == "https://example.com/a"
    assert response.content[2]["type"] == "image"
    assert response.content[2]["mediaType"] == "image/png"
