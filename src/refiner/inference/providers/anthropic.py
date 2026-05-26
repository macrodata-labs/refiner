from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from refiner.inference._message_conversion import convert_to_anthropic_payload
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference.types import InferenceWarning, Message, ProviderOptions

PROVIDER_OPTIONS = {
    "sendReasoning",
    "structuredOutputMode",
    "thinking",
    "disableParallelToolUse",
    "cacheControl",
    "metadata",
    "mcpServers",
    "container",
    "serviceTier",
    "toolStreaming",
    "effort",
    "taskBudget",
    "speed",
    "inferenceGeo",
    "anthropicBeta",
    "contextManagement",
}


def build_payload(
    *,
    messages: list[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = convert_to_anthropic_payload(
        messages,
        params=_params(params),
        provider_options=provider_options,
    )
    if schema is not None:
        _apply_schema_instruction(payload, schema)
    return payload


def schema_warnings(schema: StructuredOutputSchema | None) -> list[InferenceWarning]:
    if schema is None:
        return []
    return [
        {
            "type": "unsupported-setting",
            "setting": "schema",
            "message": (
                "AnthropicEndpointProvider does not enforce schema natively; "
                "Refiner adds a JSON instruction and validates the response locally."
            ),
        }
    ]


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    if "maxOutputTokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("maxOutputTokens")
    return payload


def _apply_schema_instruction(
    payload: dict[str, Any],
    schema: StructuredOutputSchema,
) -> None:
    instruction = (
        "Return only valid JSON that matches this JSON Schema. "
        "Do not include markdown fences or extra prose.\n"
        f"{json.dumps(schema.json_schema, separators=(',', ':'))}"
    )
    existing = payload.get("system")
    if isinstance(existing, str) and existing:
        payload["system"] = f"{existing}\n\n{instruction}"
    else:
        payload["system"] = instruction


__all__ = ["PROVIDER_OPTIONS", "build_payload", "schema_warnings"]
