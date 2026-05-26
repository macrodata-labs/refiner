from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from refiner.inference._message_conversion import convert_to_google_payload
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference.types import Message, ProviderOptions

PROVIDER_OPTIONS = {
    "responseModalities",
    "thinkingConfig",
    "cachedContent",
    "structuredOutputs",
    "safetySettings",
    "threshold",
    "audioTimestamp",
    "labels",
    "mediaResolution",
    "imageConfig",
    "retrievalConfig",
    "streamFunctionCallArguments",
    "serviceTier",
    "sharedRequestType",
    "requestType",
}


def build_payload(
    *,
    messages: list[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = convert_to_google_payload(
        messages,
        generation_config=_generation_config(params),
        provider_options=provider_options,
    )
    if schema is not None:
        generation_config = dict(payload.get("generationConfig", {}))
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema.json_schema
        payload["generationConfig"] = generation_config
    return payload


def _generation_config(params: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(params)
    aliases = {
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "stop_sequences": "stopSequences",
    }
    for source, target in aliases.items():
        if source in config and target not in config:
            config[target] = config.pop(source)
    return config


__all__ = ["PROVIDER_OPTIONS", "build_payload"]
