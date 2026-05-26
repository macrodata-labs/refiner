from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from refiner.inference._message_conversion import (
    convert_to_openai_chat_messages,
    convert_to_openai_responses_input,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference.types import Message, ProviderOptions

CHAT_PROVIDER_OPTIONS = {
    "audio",
    "background",
    "logitBias",
    "logprobs",
    "modalities",
    "parallelToolCalls",
    "user",
    "responseFormat",
    "reasoningEffort",
    "maxCompletionTokens",
    "store",
    "metadata",
    "prediction",
    "serviceTier",
    "reasoningSummary",
    "textVerbosity",
    "promptCacheKey",
    "promptCacheRetention",
    "safetyIdentifier",
    "text",
    "topLogprobs",
    "webSearchOptions",
}

RESPONSES_PROVIDER_OPTIONS = {
    *CHAT_PROVIDER_OPTIONS,
    "conversation",
    "include",
    "instructions",
    "maxToolCalls",
    "previousResponseId",
    "truncation",
    "contextManagement",
}


def build_chat_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["messages"] = convert_to_openai_chat_messages(messages)
    elif schema is None:
        payload["prompt"] = prompt
    else:
        payload["messages"] = [{"role": "user", "content": prompt or ""}]
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": _json_schema(schema),
        }
    if provider_options is not None:
        payload["providerOptions"] = provider_options
    return payload


def build_responses_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["input"] = convert_to_openai_responses_input(messages)
    else:
        payload["input"] = prompt
    _apply_responses_options(payload, provider_options)
    _normalize_responses_params(payload)
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        text = dict(payload.get("text", {}))
        text["format"] = _response_format(schema)
        payload["text"] = text
    return payload


def _apply_responses_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    passthrough = {
        "conversation",
        "background",
        "include",
        "instructions",
        "logprobs",
        "maxToolCalls",
        "metadata",
        "parallelToolCalls",
        "previousResponseId",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "serviceTier",
        "store",
        "truncation",
        "user",
        "contextManagement",
        "text",
        "topLogprobs",
        "webSearchOptions",
    }
    for key in passthrough:
        if key in openai_options:
            payload[key] = openai_options[key]


def _normalize_responses_params(payload: dict[str, Any]) -> None:
    aliases = {
        "max_tokens": "max_output_tokens",
        "maxCompletionTokens": "max_output_tokens",
    }
    for source, target in aliases.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)


def _normalize_reasoning_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    for key in (
        "logitBias",
        "logprobs",
        "topLogprobs",
        "parallelToolCalls",
        "user",
        "maxCompletionTokens",
        "modalities",
        "audio",
        "store",
        "metadata",
        "prediction",
        "serviceTier",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "webSearchOptions",
        "responseFormat",
    ):
        if key in openai_options:
            payload[key] = openai_options[key]
    reasoning: dict[str, Any] = {}
    if "reasoningEffort" in openai_options:
        reasoning["effort"] = openai_options["reasoningEffort"]
    if "reasoningSummary" in openai_options:
        reasoning["summary"] = openai_options["reasoningSummary"]
    if reasoning:
        payload["reasoning"] = {**dict(payload.get("reasoning", {})), **reasoning}


def _normalize_text_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    if "textVerbosity" in openai_options:
        text = dict(payload.get("text", {}))
        text["verbosity"] = openai_options["textVerbosity"]
        payload["text"] = text


def _response_format(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


def _json_schema(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


__all__ = [
    "CHAT_PROVIDER_OPTIONS",
    "RESPONSES_PROVIDER_OPTIONS",
    "build_chat_payload",
    "build_responses_payload",
]
