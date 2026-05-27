from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import pytest

import refiner as mdr
from refiner.inference import (
    GoogleEndpointProvider,
    InferenceResponse,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    google_provider,
)


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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
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
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
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
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
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
                            "providerOptions": {
                                "google": {"thoughtSignature": "image-sig"}
                            },
                        },
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
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
                        },
                        "thoughtSignature": "image-sig",
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
                        "content": {
                            "parts": [
                                {
                                    "text": "think",
                                    "thought": True,
                                    "thoughtSignature": "vertex-response-sig",
                                },
                                {"text": "ok"},
                            ]
                        },
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

    monkeypatch.setattr(google_provider.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "secret")

    response = asyncio.run(
        google_provider._GoogleEndpointClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.5-flash",
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


def test_google_endpoint_client_passes_vertex_request_headers(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, object]:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "think",
                                    "thought": True,
                                    "thoughtSignature": "vertex-response-sig",
                                },
                                {"text": "ok"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            }

    class _FakeAsyncClient:
        def __init__(self, *, base_url, headers, timeout):
            seen["base_url"] = str(base_url)
            seen["client_headers"] = dict(headers)
            seen["timeout"] = timeout

        async def post(self, path, *, json, headers):
            seen["path"] = path
            seen["payload"] = dict(json)
            seen["headers"] = dict(headers)
            return _FakeResponse()

    monkeypatch.setattr(google_provider.httpx, "AsyncClient", _FakeAsyncClient)

    payload = google_provider.build_payload(
        messages=[{"role": "user", "content": "hello"}],
        params={},
        provider_options={
            "googleVertex": {
                "sharedRequestType": "priority",
                "requestType": "shared",
            }
        },
        schema=None,
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
    )

    response = asyncio.run(
        google_provider._GoogleEndpointClient(
            base_url="https://us-central1-aiplatform.googleapis.com/v1",
            model="publishers/google/models/gemini-2.5-flash",
        ).generate_text(payload)
    )

    assert response.text == "ok"
    assert response.content[0] == {
        "type": "reasoning",
        "text": "think",
        "providerMetadata": {
            "googleVertex": {
                "thought": True,
                "thoughtSignature": "vertex-response-sig",
            }
        },
    }
    assert seen["headers"] == {
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
        "X-Vertex-AI-LLM-Request-Type": "shared",
    }
    assert "__refiner_headers" not in cast(Mapping[str, object], seen["payload"])


def test_google_vertex_payload_reads_vertex_thought_signatures() -> None:
    payload = google_provider.build_payload(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "text": "Previous thinking.",
                        "providerOptions": {
                            "googleVertex": {"thoughtSignature": "vertex-sig"}
                        },
                    }
                ],
            }
        ],
        params={},
        provider_options=None,
        schema=None,
        base_url="https://us-central1-aiplatform.googleapis.com/v1",
    )

    assert payload == {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "text": "Previous thinking.",
                        "thought": True,
                        "thoughtSignature": "vertex-sig",
                    }
                ],
            }
        ]
    }


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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        return {
            "output": (
                await generate_text(
                    messages=[
                        {"role": "user", "content": "Generate an image caption."}
                    ],
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
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
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


def test_parse_google_response_includes_reasoning_content() -> None:
    response = google_provider.parse_response(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "think",
                                "thought": True,
                                "thoughtSignature": "sig",
                            },
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
        {
            "type": "reasoning",
            "text": "think",
            "providerMetadata": {
                "google": {"thoughtSignature": "sig", "thought": True}
            },
        },
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


def test_parse_google_response_reports_prompt_block_reason() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "google generation response is missing candidates\\[0\\]: "
            "promptFeedback.blockReason=PROHIBITED_CONTENT"
        ),
    ):
        google_provider.parse_response(
            {
                "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                "usageMetadata": {"promptTokenCount": 1},
            }
        )
