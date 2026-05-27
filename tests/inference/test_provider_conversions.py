from __future__ import annotations

from typing import Any, cast

from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider


def test_openai_chat_conversion_matches_provider_wire_shape() -> None:
    messages = cast(
        list[Any],
        [
            {"role": "system", "content": "Use short answers."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "file",
                        "mediaType": "image/png",
                        "data": b"\x89PNG\r\n\x1a\nimage",
                        "providerOptions": {"openai": {"imageDetail": "low"}},
                    },
                ],
            },
        ],
    )

    payload = openai_provider.build_chat_payload(
        messages=messages,
        params={"model": "gpt-test", "temperature": 0},
        provider_options=None,
        schema=None,
    )

    assert payload == {
        "model": "gpt-test",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Use short answers."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgppbWFnZQ==",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
    }


def test_openai_responses_conversion_matches_provider_wire_shape() -> None:
    messages = cast(
        list[Any],
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "Previous reasoning."},
                    {"type": "text", "text": "Previous answer."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this PDF."},
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "filename": "paper.pdf",
                        "data": b"pdf-bytes",
                    },
                ],
            },
        ],
    )

    payload = openai_provider.build_responses_payload(
        messages=messages,
        params={"model": "gpt-5-mini"},
        provider_options={"openai": {"reasoningEffort": "low"}},
        schema=None,
    )

    assert payload == {
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
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Read this PDF."},
                    {
                        "type": "input_file",
                        "filename": "paper.pdf",
                        "file_data": "data:application/pdf;base64,cGRmLWJ5dGVz",
                    },
                ],
            },
        ],
        "reasoning": {"effort": "low"},
    }


def test_google_conversion_matches_provider_wire_shape() -> None:
    messages = cast(
        list[Any],
        [
            {"role": "system", "content": "You are a video annotator."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize."},
                    {
                        "type": "file",
                        "mediaType": "video",
                        "data": b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00",
                    },
                ],
            },
        ],
    )

    payload = google_provider.build_payload(
        messages=messages,
        params={"temperature": 0.1},
        provider_options={"google": {"responseModalities": ["TEXT"]}},
        schema=None,
    )

    assert payload == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Summarize."},
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": "AAAAGGZ0eXBpc29tAAAAAA==",
                        }
                    },
                ],
            }
        ],
        "systemInstruction": {"parts": [{"text": "You are a video annotator."}]},
        "generationConfig": {
            "temperature": 0.1,
            "responseModalities": ["TEXT"],
        },
    }


def test_anthropic_conversion_matches_provider_wire_shape() -> None:
    messages = cast(
        list[Any],
        [
            {"role": "system", "content": "Use citations."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document."},
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "data": b"pdf-bytes",
                        "providerOptions": {
                            "anthropic": {"citations": {"enabled": True}}
                        },
                    },
                ],
            },
        ],
    )

    payload = anthropic_provider.build_payload(
        messages=messages,
        params={"model": "claude-test", "max_tokens": 256},
        provider_options=None,
        schema=None,
    )

    assert payload == {
        "model": "claude-test",
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
                        "title": "part-1",
                        "citations": {"enabled": True},
                    },
                ],
            }
        ],
        "system": [{"type": "text", "text": "Use citations."}],
    }
