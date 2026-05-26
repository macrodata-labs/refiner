from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, cast

from pydantic import BaseModel

from refiner.inference.capabilities import capability_warnings
from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider
from refiner.inference.providers.warnings import provider_option_warnings
from refiner.inference._runtime import RequestFn, inference_map
from refiner.inference._schema import (
    normalize_schema,
    validate_structured_output,
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
        schema: type[BaseModel] | None = None,
        schemaStrict: bool = True,
        **params: Any,
    ) -> Awaitable[InferenceResponse]: ...


GenerateTextMapFn = Callable[[Row, GenerateTextFn], Awaitable[MapResult] | MapResult]


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
            schema: type[BaseModel] | None = None,
            schemaStrict: bool = True,
            **params: Any,
        ) -> InferenceResponse:
            if (messages is None) == (prompt is None):
                raise ValueError("pass exactly one of messages or prompt")
            if maxRetries is not None and max_retries is not None:
                raise ValueError("pass only one of maxRetries or max_retries")
            schema_info = normalize_schema(
                schema,
                strict=schemaStrict,
            )
            payload = {**dict(default_generation_params or {}), **params}
            retry_override = maxRetries if maxRetries is not None else max_retries
            warnings = _provider_warnings(provider, providerOptions)
            request_messages: Sequence[Message]
            if messages is not None:
                request_messages = list(messages)
            else:
                request_messages = [{"role": "user", "content": prompt or ""}]
            warnings.extend(
                capability_warnings(
                    provider=provider,
                    messages=request_messages,
                    params=payload,
                    provider_options=providerOptions,
                    has_schema=schema_info is not None,
                )
            )
            if messages is not None:
                if isinstance(provider, GoogleEndpointProvider):
                    payload = google_provider.build_payload(
                        messages=list(messages),
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                elif isinstance(provider, AnthropicEndpointProvider):
                    warnings.extend(anthropic_provider.schema_warnings(schema_info))
                    payload = anthropic_provider.build_payload(
                        messages=list(messages),
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                elif isinstance(provider, OpenAIResponsesProvider):
                    payload = openai_provider.build_responses_payload(
                        messages=messages,
                        prompt=None,
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                else:
                    payload = openai_provider.build_chat_payload(
                        messages=messages,
                        prompt=None,
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
            else:
                user_message: Message = {"role": "user", "content": prompt or ""}
                if isinstance(provider, GoogleEndpointProvider):
                    payload = google_provider.build_payload(
                        messages=[user_message],
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                elif isinstance(provider, AnthropicEndpointProvider):
                    warnings.extend(anthropic_provider.schema_warnings(schema_info))
                    payload = anthropic_provider.build_payload(
                        messages=[user_message],
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                elif isinstance(provider, OpenAIResponsesProvider):
                    payload = openai_provider.build_responses_payload(
                        messages=None,
                        prompt=prompt,
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
                else:
                    payload = openai_provider.build_chat_payload(
                        messages=None,
                        prompt=prompt,
                        params=payload,
                        provider_options=providerOptions,
                        schema=schema_info,
                    )
            if retry_override is not None:
                payload["__refiner_max_retries"] = retry_override
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


def _provider_warnings(
    provider: (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    ),
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
