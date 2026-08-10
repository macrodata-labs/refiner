from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest
from pydantic import BaseModel

import refiner as mdr
from refiner.inference import (
    AnthropicEndpointProvider,
    GoogleEndpointProvider,
    InferenceResponse,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
)
from refiner.pipeline.data.row import DictRow

from ._helpers import (
    _Caption,
    _ConstrainedCaption,
    _Segments,
    anthropic_provider,
    google_provider,
    openai_provider,
)


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

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "caption"}], schema=_Caption
        )
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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "caption"}], schema=_Caption
        )
        assert isinstance(response.object, _Caption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"title": "Video"}
    payload = cast(Mapping[str, object], seen["payload"])
    generation_config = cast(Mapping[str, object], payload["generationConfig"])
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseSchema"] == {
        "properties": {
            "title": {"type": "string"},
            "objects": {"items": {"type": "string"}, "type": "array"},
        },
        "required": ["title", "objects"],
        "type": "object",
    }


def test_inference_generate_text_preserves_google_schema_constraints(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text='{"title":"Video","score":5,"objects":["car"]}',
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
            messages=[{"role": "user", "content": "caption"}],
            schema=_ConstrainedCaption,
        )
        assert isinstance(response.object, _ConstrainedCaption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"title": "Video"}
    payload = cast(Mapping[str, object], seen["payload"])
    generation_config = cast(Mapping[str, object], payload["generationConfig"])
    response_schema = cast(Mapping[str, object], generation_config["responseSchema"])
    assert response_schema["properties"] == {
        "title": {"maxLength": 20, "minLength": 2, "type": "string"},
        "score": {"maximum": 5, "minimum": 0, "type": "integer"},
        "objects": {
            "items": {"type": "string"},
            "maxItems": 3,
            "minItems": 1,
            "type": "array",
        },
    }


def test_inference_generate_text_inlines_google_schema_refs(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_generate_text(self, payload):
        del self
        seen["payload"] = dict(payload)
        return InferenceResponse(
            text=(
                '{"segments":[{"start_sec":0.0,"end_sec":4.0,"subtask":"open drawer"}]}'
            ),
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
            messages=[{"role": "user", "content": "segment"}],
            schema=_Segments,
        )
        assert isinstance(response.object, _Segments)
        return {"subtask": response.object.segments[0].subtask}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"subtask": "open drawer"}
    payload = cast(Mapping[str, object], seen["payload"])
    generation_config = cast(Mapping[str, object], payload["generationConfig"])
    response_schema = cast(Mapping[str, object], generation_config["responseSchema"])
    assert "$defs" not in response_schema
    assert "$ref" not in json.dumps(response_schema)
    assert response_schema == {
        "properties": {
            "segments": {
                "items": {
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "subtask": {"type": "string"},
                    },
                    "required": ["start_sec", "end_sec", "subtask"],
                    "type": "object",
                },
                "type": "array",
            }
        },
        "required": ["segments"],
        "type": "object",
    }


def test_inference_generate_text_rejects_cyclic_google_schema_refs(
    monkeypatch,
) -> None:
    class _Node(BaseModel):
        name: str
        child: _Node | None = None

    async def _fake_generate_text(self, payload):
        del self, payload
        raise AssertionError("cyclic schema should fail before provider request")

    monkeypatch.setattr(
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            messages=[{"role": "user", "content": "tree"}],
            schema=_Node,
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
    )

    with pytest.raises(
        ValueError,
        match="cyclic structured output schema refs are not supported",
    ):
        asyncio.run(cast(Any, infer(DictRow({}))))


def test_inference_generate_text_can_disable_google_structured_outputs(
    monkeypatch,
) -> None:
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
        google_provider._GoogleEndpointClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "caption"}],
            schema=_Caption,
            provider_options={"google": {"structuredOutputs": False}},
        )
        assert isinstance(response.object, _Caption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=GoogleEndpointProvider(model="gemini-2.5-flash"),
    )

    assert asyncio.run(cast(Any, infer(DictRow({})))) == {"title": "Video"}
    payload = cast(Mapping[str, object], seen["payload"])
    generation_config = cast(Mapping[str, object], payload["generationConfig"])
    assert generation_config["responseMimeType"] == "application/json"
    assert "responseSchema" not in generation_config


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
        openai_provider._OpenAIResponsesClient, "generate_text", _fake_generate_text
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "caption"}], schema=_Caption
        )
        assert isinstance(response.object, _Caption)
        return {"title": response.object.title}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIResponsesProvider(model="gpt-test"),
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
        anthropic_provider._AnthropicEndpointClient,
        "generate_text",
        _fake_generate_text,
    )

    async def _inference_fn(row, generate_text):
        del row
        response = await generate_text(
            messages=[{"role": "user", "content": "caption"}], schema=_Caption
        )
        assert isinstance(response.object, _Caption)
        return {"warnings": list(response.warnings)}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=AnthropicEndpointProvider(model="claude-test"),
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
        anthropic_provider._AnthropicEndpointClient,
        "generate_text",
        _fake_generate_text,
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
        provider=AnthropicEndpointProvider(model="claude-test"),
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

    monkeypatch.setattr(
        openai_provider._OpenAIEndpointClient, "generate", _fake_generate
    )

    async def _inference_fn(row, generate_text):
        del row
        await generate_text(
            messages=[{"role": "user", "content": "caption"}], schema=_Caption
        )
        return {}

    infer = mdr.inference.generate_text(
        fn=_inference_fn,
        provider=OpenAIEndpointProvider(
            base_url="https://api.example.com", model="gpt-test"
        ),
    )

    with pytest.raises(mdr.inference.InferenceSchemaValidationError):
        asyncio.run(cast(Any, infer(DictRow({}))))
