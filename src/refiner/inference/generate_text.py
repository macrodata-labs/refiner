from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, TypeAlias, cast

from pydantic import BaseModel

from refiner.inference.capabilities import capability_warnings
from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider
from refiner.inference.providers.warnings import provider_option_warnings
from refiner.inference.internal.runtime import RequestFn, inference_map
from refiner.inference.internal.schema import (
    StructuredOutputSchema,
    normalize_schema,
    validate_structured_output,
)
from refiner.inference.internal.response import InferenceResponse
from refiner.inference.generate import _record_usage
from refiner.inference.providers import (
    AnthropicEndpointProvider,
    GoogleEndpointProvider,
    OpenAIEndpointProvider,
    OpenAIResponsesProvider,
    VLLMProvider,
)
from refiner.inference.providers.anthropic import _AnthropicEndpointClient
from refiner.inference.providers.google import _GoogleEndpointClient
from refiner.inference.providers.openai import (
    _OpenAIEndpointClient,
    _OpenAIResponsesClient,
)
from refiner.inference.types import InferenceWarning, Message, ProviderOptions
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult

_InferenceProvider: TypeAlias = (
    AnthropicEndpointProvider
    | GoogleEndpointProvider
    | OpenAIEndpointProvider
    | OpenAIResponsesProvider
    | VLLMProvider
)


class GenerateTextFn(Protocol):
    def __call__(
        self,
        *,
        messages: Sequence[Message] | None = None,
        prompt: str | None = None,
        providerOptions: ProviderOptions | None = None,
        maxRetries: int | None = None,
        schema: type[BaseModel] | None = None,
        schemaStrict: bool = True,
        **params: Any,
    ) -> Awaitable[InferenceResponse]: ...


GenerateTextMapFn = Callable[[Row, GenerateTextFn], Awaitable[MapResult] | MapResult]


def generate_text(
    *,
    fn: GenerateTextMapFn,
    provider: _InferenceProvider,
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
            schema: type[BaseModel] | None = None,
            schemaStrict: bool = True,
            **params: Any,
        ) -> InferenceResponse:
            if (messages is None) == (prompt is None):
                raise ValueError("pass exactly one of messages or prompt")
            provider_options = providerOptions
            max_retries = maxRetries
            schema_strict = schemaStrict
            schema_info = normalize_schema(
                schema,
                strict=schema_strict,
            )
            payload = {**dict(default_generation_params or {}), **params}
            warnings = _provider_warnings(provider, provider_options)
            warnings.extend(
                capability_warnings(
                    provider=provider,
                    messages=_messages_or_prompt(messages, prompt),
                    params=payload,
                    provider_options=provider_options,
                    has_schema=schema_info is not None,
                )
            )
            if isinstance(provider, AnthropicEndpointProvider):
                warnings.extend(anthropic_provider.schema_warnings(schema_info))
            payload = _build_payload(
                provider=provider,
                messages=messages,
                prompt=prompt,
                params=payload,
                provider_options=provider_options,
                schema=schema_info,
            )
            if max_retries is not None:
                payload["__refiner_max_retries"] = max_retries
            response = cast(InferenceResponse, await request(payload))
            parsed_object = validate_structured_output(response.text, schema_info)
            if parsed_object is not None:
                response = replace(response, object=parsed_object)
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


def _build_payload(
    *,
    provider: _InferenceProvider,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    if isinstance(provider, GoogleEndpointProvider):
        return google_provider.build_payload(
            messages=_messages_or_prompt(messages, prompt),
            params=params,
            provider_options=provider_options,
            schema=schema,
        )
    if isinstance(provider, AnthropicEndpointProvider):
        return anthropic_provider.build_payload(
            messages=_messages_or_prompt(messages, prompt),
            params=params,
            provider_options=provider_options,
            schema=schema,
        )
    if isinstance(provider, OpenAIResponsesProvider):
        return openai_provider.build_responses_payload(
            messages=messages,
            prompt=prompt,
            params=params,
            provider_options=provider_options,
            schema=schema,
        )
    return openai_provider.build_chat_payload(
        messages=messages,
        prompt=prompt,
        params=params,
        provider_options=provider_options,
        schema=schema,
    )


def _messages_or_prompt(
    messages: Sequence[Message] | None,
    prompt: str | None,
) -> list[Message]:
    if messages is not None:
        return list(messages)
    return [{"role": "user", "content": prompt or ""}]


def _provider_warnings(
    provider: _InferenceProvider,
    provider_options: ProviderOptions | None,
) -> list[InferenceWarning]:
    if isinstance(provider, OpenAIResponsesProvider):
        expected_namespace = "openai"
        supported = openai_provider.RESPONSES_PROVIDER_OPTIONS
    elif isinstance(provider, OpenAIEndpointProvider | VLLMProvider):
        expected_namespace = "openai"
        supported = openai_provider.CHAT_PROVIDER_OPTIONS
    elif isinstance(provider, GoogleEndpointProvider):
        expected_namespace = "google"
        supported = google_provider.PROVIDER_OPTIONS
    else:
        expected_namespace = "anthropic"
        supported = anthropic_provider.PROVIDER_OPTIONS
    return provider_option_warnings(
        provider_name=type(provider).__name__,
        expected_namespace=expected_namespace,
        supported_options=supported,
        provider_options=provider_options,
    )


__all__ = ["GenerateTextFn", "GenerateTextMapFn", "generate_text"]
