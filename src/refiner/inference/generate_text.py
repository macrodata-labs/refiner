from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, cast

from refiner.inference._runtime import RequestFn, inference_map
from refiner.inference._message_conversion import (
    convert_to_anthropic_payload,
    convert_to_google_payload,
    convert_to_openai_chat_messages,
    convert_to_openai_responses_input,
)
from refiner.inference.client import (
    _AnthropicEndpointClient,
    InferenceResponse,
    _GoogleEndpointClient,
    _OpenAIEndpointClient,
    _OpenAIResponsesClient,
)
from refiner.inference.generate import _record_usage
from refiner.inference.providers import (
    AnthropicEndpointProvider,
    GoogleEndpointProvider,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
    VLLMProvider,
)
from refiner.inference.types import InferenceWarning, Message, ProviderOptions
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult


class GenerateTextFn(Protocol):
    def __call__(
        self,
        *,
        messages: Sequence[Message] | None = None,
        prompt: str | None = None,
        providerOptions: ProviderOptions | None = None,
        maxRetries: int | None = None,
        max_retries: int | None = None,
        **params: Any,
    ) -> Awaitable[InferenceResponse]: ...


GenerateTextMapFn = Callable[[Row, GenerateTextFn], Awaitable[MapResult] | MapResult]

_OPENAI_CHAT_PROVIDER_OPTIONS = {
    "logitBias",
    "logprobs",
    "parallelToolCalls",
    "user",
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
}

_OPENAI_RESPONSES_PROVIDER_OPTIONS = {
    *_OPENAI_CHAT_PROVIDER_OPTIONS,
    "conversation",
    "include",
    "instructions",
    "maxToolCalls",
    "previousResponseId",
    "truncation",
    "contextManagement",
}

_GOOGLE_PROVIDER_OPTIONS = {
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

_ANTHROPIC_PROVIDER_OPTIONS = {
    "sendReasoning",
    "structuredOutputMode",
    "thinking",
    "disableParallelToolUse",
    "cacheControl",
    "metadata",
    "mcpServers",
    "container",
    "toolStreaming",
    "effort",
    "taskBudget",
    "speed",
    "inferenceGeo",
    "anthropicBeta",
    "contextManagement",
}


def generate_text(
    *,
    fn: GenerateTextMapFn,
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
    default_generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    async def _map(row: Row, request: RequestFn) -> MapResult:
        async def _generate_text(
            *,
            messages: Sequence[Message] | None = None,
            prompt: str | None = None,
            providerOptions: ProviderOptions | None = None,
            maxRetries: int | None = None,
            max_retries: int | None = None,
            **params: Any,
        ) -> InferenceResponse:
            if (messages is None) == (prompt is None):
                raise ValueError("pass exactly one of messages or prompt")
            if maxRetries is not None and max_retries is not None:
                raise ValueError("pass only one of maxRetries or max_retries")
            payload = {**dict(default_generation_params or {}), **params}
            retry_override = maxRetries if maxRetries is not None else max_retries
            warnings = _provider_option_warnings(provider, providerOptions)
            if messages is not None:
                if isinstance(provider, GoogleEndpointProvider):
                    payload = convert_to_google_payload(
                        messages,
                        generation_config=_google_generation_config(payload),
                        provider_options=providerOptions,
                    )
                elif isinstance(provider, AnthropicEndpointProvider):
                    payload = convert_to_anthropic_payload(
                        messages,
                        params=_anthropic_params(payload),
                        provider_options=providerOptions,
                    )
                elif isinstance(provider, OpenAIResponsesProvider):
                    payload["input"] = convert_to_openai_responses_input(messages)
                    _apply_openai_responses_options(payload, providerOptions)
                    _normalize_openai_responses_params(payload)
                    _normalize_openai_reasoning_options(payload, providerOptions)
                    _normalize_openai_text_options(payload, providerOptions)
                else:
                    payload["messages"] = convert_to_openai_chat_messages(messages)
                    _normalize_openai_reasoning_options(payload, providerOptions)
                    _normalize_openai_text_options(payload, providerOptions)
            else:
                if isinstance(provider, GoogleEndpointProvider):
                    payload = convert_to_google_payload(
                        [{"role": "user", "content": prompt or ""}],
                        generation_config=_google_generation_config(payload),
                        provider_options=providerOptions,
                    )
                elif isinstance(provider, AnthropicEndpointProvider):
                    payload = convert_to_anthropic_payload(
                        [{"role": "user", "content": prompt or ""}],
                        params=_anthropic_params(payload),
                        provider_options=providerOptions,
                    )
                elif isinstance(provider, OpenAIResponsesProvider):
                    payload["input"] = prompt
                    _apply_openai_responses_options(payload, providerOptions)
                    _normalize_openai_responses_params(payload)
                    _normalize_openai_reasoning_options(payload, providerOptions)
                    _normalize_openai_text_options(payload, providerOptions)
                else:
                    payload["prompt"] = prompt
                    _normalize_openai_reasoning_options(payload, providerOptions)
                    _normalize_openai_text_options(payload, providerOptions)
            if providerOptions is not None and not isinstance(
                provider,
                GoogleEndpointProvider
                | AnthropicEndpointProvider
                | OpenAIResponsesProvider,
            ):
                payload["providerOptions"] = providerOptions
            if retry_override is not None:
                payload["__refiner_max_retries"] = retry_override
            response = cast(InferenceResponse, await request(payload))
            if warnings:
                return replace(
                    response,
                    warnings=(*response.warnings, *warnings),
                )
            return response

        result = fn(row, _generate_text)
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    return inference_map(
        name="inference.generate_text",
        fn=_map,
        provider=provider,
        defaults=default_generation_params,
        defaults_key="default_generation_params",
        merge_defaults=False,
        max_concurrent_requests=max_concurrent_requests,
        call=_generate,
        record=_record_usage,
    )


async def _generate(
    client: (
        _AnthropicEndpointClient
        | _GoogleEndpointClient
        | _OpenAIEndpointClient
        | _OpenAIResponsesClient
    ),
    payload: Mapping[str, Any],
) -> InferenceResponse:
    if isinstance(
        client,
        _AnthropicEndpointClient | _GoogleEndpointClient | _OpenAIResponsesClient,
    ):
        return await client.generate_text(payload)
    return await client.generate(payload)


def _google_generation_config(params: Mapping[str, Any]) -> dict[str, Any]:
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


def _provider_option_warnings(
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
    provider_options: ProviderOptions | None,
) -> list[InferenceWarning]:
    if not provider_options:
        return []
    expected_namespace = _provider_option_namespace(provider)
    warnings: list[InferenceWarning] = []
    for namespace in provider_options:
        if namespace != expected_namespace:
            warnings.append(
                {
                    "type": "unsupported-provider-option",
                    "setting": f"providerOptions.{namespace}",
                    "message": (
                        f"{namespace!r} provider options are not used by "
                        f"{type(provider).__name__}."
                    ),
                }
            )
    expected_options = provider_options.get(expected_namespace, {})
    if isinstance(provider, OpenAIResponsesProvider):
        supported = _OPENAI_RESPONSES_PROVIDER_OPTIONS
    elif isinstance(provider, OpenAIEndpointProvider | VLLMProvider):
        supported = _OPENAI_CHAT_PROVIDER_OPTIONS
    elif isinstance(provider, GoogleEndpointProvider):
        supported = _GOOGLE_PROVIDER_OPTIONS
    else:
        supported = _ANTHROPIC_PROVIDER_OPTIONS
    for option in expected_options:
        if option not in supported:
            warnings.append(
                {
                    "type": "unsupported-setting",
                    "setting": f"providerOptions.{expected_namespace}.{option}",
                    "message": (
                        f"{option!r} is not currently mapped by "
                        f"{type(provider).__name__}."
                    ),
                }
            )
    return warnings


def _provider_option_namespace(
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
) -> str:
    if isinstance(provider, GoogleEndpointProvider):
        return "google"
    if isinstance(provider, AnthropicEndpointProvider):
        return "anthropic"
    return "openai"


def _anthropic_params(params: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    if "maxOutputTokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("maxOutputTokens")
    return payload


def _apply_openai_responses_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    passthrough = {
        "conversation",
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
    }
    for key in passthrough:
        if key in openai_options:
            payload[key] = openai_options[key]


def _normalize_openai_responses_params(payload: dict[str, Any]) -> None:
    aliases = {
        "max_tokens": "max_output_tokens",
        "maxCompletionTokens": "max_output_tokens",
    }
    for source, target in aliases.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)


def _normalize_openai_reasoning_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    for key in (
        "logitBias",
        "logprobs",
        "parallelToolCalls",
        "user",
        "maxCompletionTokens",
        "store",
        "metadata",
        "prediction",
        "serviceTier",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
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


def _normalize_openai_text_options(
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


__all__ = ["GenerateTextFn", "GenerateTextMapFn", "generate_text"]
