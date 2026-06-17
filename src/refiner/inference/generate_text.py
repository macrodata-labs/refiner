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
from refiner.inference.internal.runtime import RequestFn, inference_map
from refiner.inference.internal.schema import (
    StructuredOutputSchema,
    normalize_schema,
    validate_structured_output,
)
from refiner.inference.internal.response import InferenceResponse
from refiner.inference.internal.usage import record_usage
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
from refiner.inference.types import (
    InferenceProvider,
    InferenceWarning,
    Message,
    ProviderOptions,
)
from refiner.pipeline.data.row import Row
from refiner.pipeline.builtins import describe_builtin
from refiner.pipeline.steps import MapResult


class GenerateTextFn(Protocol):
    def __call__(
        self,
        *,
        messages: Sequence[Message] | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        provider_options: ProviderOptions | None = None,
        maxRetries: int | None = None,
        schema: type[BaseModel] | None = None,
        schemaStrict: bool = True,
        **params: Any,
    ) -> Awaitable[InferenceResponse]: ...


GenerateTextMapFn = Callable[[Row, GenerateTextFn], Awaitable[MapResult] | MapResult]


def generate_text(
    *,
    fn: GenerateTextMapFn,
    provider: InferenceProvider,
    default_generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    """Return an async row mapper that issues text-generation requests.

    Args:
        fn: Row-level function that receives the input row and a request function.
            The request function accepts either typed messages or a raw provider
            payload and returns an ``InferenceResponse``.
        provider: Endpoint or runtime-service provider used to execute requests.
        default_generation_params: Parameters merged into each typed message or
            raw payload request unless overridden by that individual request.
        max_concurrent_requests: Maximum number of provider requests allowed to
            run at once per worker.
    """

    @describe_builtin(
        "inference.generate_text",
        fn=fn,
        provider=provider.to_builtin_args(),
        max_concurrent_requests=max_concurrent_requests,
        default_generation_params=dict(default_generation_params or {}),
    )
    async def _map(row: Row, request: RequestFn) -> MapResult:
        async def _generate_text(
            *,
            messages: Sequence[Message] | None = None,
            raw_payload: Mapping[str, Any] | None = None,
            provider_options: ProviderOptions | None = None,
            maxRetries: int | None = None,
            schema: type[BaseModel] | None = None,
            schemaStrict: bool = True,
            **params: Any,
        ) -> InferenceResponse:
            if (messages is None) == (raw_payload is None):
                raise ValueError("pass exactly one of messages or raw_payload")
            max_retries = maxRetries
            schema_strict = schemaStrict
            schema_info = normalize_schema(
                schema,
                strict=schema_strict,
            )
            default_params = dict(default_generation_params or {})
            payload = {**default_params, **params}
            if raw_payload is not None:
                if provider_options is not None:
                    raise ValueError(
                        "provider_options are not supported with raw_payload"
                    )
                if schema is not None:
                    raise ValueError("schema is not supported with raw_payload")
                payload = {**default_params, **dict(raw_payload), **params}
                if max_retries is not None:
                    payload["__refiner_max_retries"] = max_retries
                return cast(InferenceResponse, await request(payload))

            typed_messages = cast(Sequence[Message], messages)
            warnings = _provider_warnings(provider, provider_options)
            warnings.extend(
                capability_warnings(
                    provider=provider,
                    messages=typed_messages,
                    params=payload,
                    provider_options=provider_options,
                    has_schema=schema_info is not None,
                )
            )
            if isinstance(provider, AnthropicEndpointProvider):
                warnings.extend(anthropic_provider.schema_warnings(schema_info))
            payload = _build_payload(
                provider=provider,
                messages=typed_messages,
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
        record=record_usage,
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
    provider: InferenceProvider,
    messages: Sequence[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    if isinstance(provider, GoogleEndpointProvider):
        return google_provider.build_payload(
            messages=messages,
            params=params,
            provider_options=provider_options,
            schema=schema,
            base_url=provider.base_url,
        )
    if isinstance(provider, AnthropicEndpointProvider):
        return anthropic_provider.build_payload(
            messages=messages,
            params=params,
            provider_options=provider_options,
            schema=schema,
        )
    if isinstance(provider, OpenAIResponsesProvider):
        return openai_provider.build_responses_payload(
            messages=messages,
            params=params,
            provider_options=provider_options,
            schema=schema,
        )
    return openai_provider.build_chat_payload(
        messages=messages,
        params=params,
        provider_options=provider_options,
        schema=schema,
    )


def _provider_warnings(
    provider: InferenceProvider,
    provider_options: ProviderOptions | None,
) -> list[InferenceWarning]:
    if isinstance(provider, OpenAIResponsesProvider):
        expected_namespace = "openai"
        supported = openai_provider.RESPONSES_PROVIDER_OPTIONS
    elif isinstance(provider, OpenAIEndpointProvider | VLLMProvider):
        expected_namespace = "openai"
        supported = openai_provider.CHAT_PROVIDER_OPTIONS
    elif isinstance(provider, GoogleEndpointProvider):
        expected_namespace = (
            ("googleVertex", "vertex", "google")
            if google_provider.is_vertex_base_url(provider.base_url)
            else "google"
        )
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
