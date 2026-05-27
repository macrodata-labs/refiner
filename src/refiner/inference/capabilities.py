from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from refiner.inference.internal.media import resolve_media_type
from refiner.inference.providers import (
    AnthropicEndpointProvider,
    GoogleEndpointProvider,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
    VLLMProvider,
)
from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider
from refiner.inference.types import InferenceWarning, Message, ModelCapabilities

_MEDIA_WARNING_BYTES = 20 * 1024 * 1024
_TOOL_SETTINGS = {
    "tools",
    "tool_choice",
    "toolChoice",
    "parallelToolCalls",
    "maxToolCalls",
}


def model_capabilities(
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
) -> ModelCapabilities:
    model = provider.model.lower()
    if isinstance(provider, GoogleEndpointProvider):
        return google_provider.model_capabilities(model)
    if isinstance(provider, AnthropicEndpointProvider):
        return anthropic_provider.model_capabilities(model)
    if isinstance(provider, OpenAIResponsesProvider):
        return openai_provider.model_capabilities(model, responses_api=True)
    if isinstance(provider, OpenAIEndpointProvider):
        return openai_provider.model_capabilities(model, responses_api=False)
    return ModelCapabilities()


def capability_warnings(
    *,
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
    messages: Sequence[Message],
    params: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None,
    has_schema: bool,
) -> list[InferenceWarning]:
    capabilities = model_capabilities(provider)
    warnings: list[InferenceWarning] = []
    provider_name = type(provider).__name__

    if (
        has_schema
        and capabilities.structured_output is False
        and not isinstance(provider, AnthropicEndpointProvider)
    ):
        warnings.append(
            {
                "type": "unsupported-setting",
                "setting": "schema",
                "message": (
                    f"{provider_name} model {provider.model!r} is not known to "
                    "support native structured output; Refiner may validate locally."
                ),
            }
        )

    warnings.extend(
        _model_setting_warnings(
            provider=provider,
            params=params,
            provider_options=provider_options,
        )
    )

    for setting in sorted(_TOOL_SETTINGS):
        if setting in params or _has_provider_option(provider_options, setting):
            warnings.append(
                {
                    "type": "unsupported-setting",
                    "setting": setting,
                    "message": (
                        f"{provider_name} received {setting!r}, but Refiner "
                        "generate_text does not implement tool calling yet."
                    ),
                }
            )

    for media_type, size, message_index, part_index in _message_media(messages):
        top_level = media_type.split("/", 1)[0]
        supported = _media_supported(capabilities, top_level, media_type)
        if supported is False:
            warnings.append(
                {
                    "type": "unsupported-content",
                    "setting": f"messages[{message_index}].content[{part_index}]",
                    "message": (
                        f"{provider_name} model {provider.model!r} is not known "
                        f"to support {top_level} input."
                    ),
                    "details": media_type,
                }
            )
        if size is not None and size > _MEDIA_WARNING_BYTES:
            warnings.append(
                {
                    "type": "unsupported-content",
                    "setting": f"messages[{message_index}].content[{part_index}]",
                    "message": (
                        "inline media is larger than 20 MiB; provider APIs may "
                        "reject it or require a file upload/URI path."
                    ),
                    "details": f"{media_type}; {size} bytes",
                }
            )
    return warnings


def _model_setting_warnings(
    *,
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
    params: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None,
) -> list[InferenceWarning]:
    if isinstance(provider, OpenAIEndpointProvider | OpenAIResponsesProvider):
        return openai_provider.model_setting_warnings(
            model=provider.model,
            provider_name=type(provider).__name__,
            responses_api=isinstance(provider, OpenAIResponsesProvider),
            params=params,
            provider_options=provider_options,
        )
    if isinstance(provider, AnthropicEndpointProvider):
        return anthropic_provider.model_setting_warnings(
            model=provider.model,
            params=params,
            provider_options=provider_options,
        )
    return []


def _has_provider_option(
    provider_options: Mapping[str, Mapping[str, Any]] | None, setting: str
) -> bool:
    if provider_options is None:
        return False
    return any(setting in options for options in provider_options.values())


def _media_supported(
    capabilities: ModelCapabilities, top_level: str, media_type: str
) -> bool | None:
    if top_level == "image":
        return capabilities.images
    if top_level == "audio":
        return capabilities.audio
    if top_level == "video":
        return capabilities.video
    if media_type == "application/pdf" or top_level == "text":
        return capabilities.files
    return capabilities.files


def _message_media(
    messages: Sequence[Message],
) -> list[tuple[str, int | None, int, int]]:
    media: list[tuple[str, int | None, int, int]] = []
    for message_index, message in enumerate(messages):
        content = message["content"]
        if isinstance(content, str):
            continue
        for part_index, part in enumerate(content):
            part_type = part.get("type")
            if part_type == "image":
                data = part.get("image")
                media_type = resolve_media_type(
                    data,
                    declared_media_type=part.get("mediaType"),
                    default_top_level="image",
                )
            elif part_type == "file":
                data = part.get("data")
                media_type = resolve_media_type(
                    data,
                    declared_media_type=part.get("mediaType"),
                )
            else:
                continue
            media.append((media_type, _data_size(data), message_index, part_index))
    return media


def _data_size(data: object) -> int | None:
    if isinstance(data, bytes | bytearray | memoryview):
        return len(data)
    if not isinstance(data, str):
        return None
    marker = ";base64,"
    if marker not in data:
        return None
    encoded = data.split(marker, 1)[1]
    try:
        return len(base64.b64decode(encoded, validate=True))
    except ValueError:
        return None


__all__ = ["ModelCapabilities", "capability_warnings", "model_capabilities"]
