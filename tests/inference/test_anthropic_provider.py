from __future__ import annotations

import asyncio


import refiner as mdr
from refiner.inference import (
    AnthropicEndpointProvider,
    InferenceResponse,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    anthropic_provider,
)


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
        anthropic_provider._AnthropicEndpointClient,
        "generate_text",
        _fake_generate_text,
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
        provider=AnthropicEndpointProvider(model="claude-sonnet-4-5"),
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
        anthropic_provider._AnthropicEndpointClient,
        "generate_text",
        _fake_generate_text,
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "Previous reasoning."},
                        {
                            "type": "reasoning",
                            "text": "Signed reasoning.",
                            "providerOptions": {
                                "anthropic": {"signature": "thinking-sig"}
                            },
                        },
                        {"type": "text", "text": "Previous answer.   "},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        )
        return {"output": response.text}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-sonnet-4-5"),
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
                    {
                        "type": "thinking",
                        "thinking": "Signed reasoning.",
                        "signature": "thinking-sig",
                    },
                    {"type": "text", "text": "Previous answer."},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
        ],
    }


def test_parse_anthropic_response_includes_reasoning_content() -> None:
    response = anthropic_provider.parse_response(
        {
            "content": [
                {"type": "thinking", "thinking": "think", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
            "id": "msg_123",
            "model": "claude-sonnet-4-5",
            "usage": {},
            "stop_reason": "end_turn",
        }
    )

    assert response.text == "answer"
    assert list(response.content) == [
        {
            "type": "reasoning",
            "text": "think",
            "providerMetadata": {"anthropic": {"type": "thinking", "signature": "sig"}},
        },
        {"type": "text", "text": "answer"},
    ]
    assert response.provider_metadata == {
        "anthropic": {
            "id": "msg_123",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {},
        }
    }


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
