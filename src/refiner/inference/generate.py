from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, cast

from refiner.inference.client import InferenceResponse, _OpenAIEndpointClient
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.pipeline.builtins import REFINER_BUILTIN_CALL_ATTR
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_binding

GeneratePayload: TypeAlias = Mapping[str, Any]
GenerateFn: TypeAlias = Callable[[GeneratePayload], Awaitable[InferenceResponse]]
InferenceFn: TypeAlias = Callable[
    [Row, GenerateFn],
    Awaitable[MapResult] | MapResult,
]


def generate(
    *,
    fn: InferenceFn,
    provider: OpenAIEndpointProvider | VLLMProvider,
    default_generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 128,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")

    request_semaphore = asyncio.Semaphore(max_concurrent_requests)
    client: _OpenAIEndpointClient | None = None
    client_lock = asyncio.Lock()

    async def _generate(payload: GeneratePayload) -> InferenceResponse:
        nonlocal client
        request_payload = dict(default_generation_params or {})
        request_payload.update(dict(payload))
        if client is None:
            async with client_lock:
                if client is None:
                    if isinstance(provider, OpenAIEndpointProvider):
                        client = _OpenAIEndpointClient(
                            base_url=provider.base_url,
                            api_key=provider.api_key,
                        )
                    else:
                        service_name = provider.service_definition().name
                        binding = get_active_service_binding(service_name)
                        if binding is None:
                            raise RuntimeError(
                                f"VLLM provider requires runtime service binding {service_name!r}"
                            )
                        if not isinstance(binding, VLLMRuntimeServiceBinding):
                            raise RuntimeError(
                                f"VLLM provider expected a VLLM runtime binding for {service_name!r}"
                            )
                        client = _OpenAIEndpointClient(
                            base_url=binding.endpoint,
                            api_key=binding.api_key,
                        )
            assert client is not None

        async with request_semaphore:
            return await client.generate(request_payload)

    async def _wrapped(row: Row) -> MapResult:
        result = fn(row, _generate)
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    setattr(
        _wrapped,
        REFINER_BUILTIN_CALL_ATTR,
        {
            "name": "inference.generate",
            "args": {
                "fn": fn,
                "provider": provider.to_builtin_args(),
                "default_generation_params": dict(default_generation_params or {}),
                "max_concurrent_requests": max_concurrent_requests,
            },
            "services": [
                service_definition.to_spec()
                for service_definition in [provider.service_definition()]
                if service_definition is not None
            ],
        },
    )
    return _wrapped


__all__ = ["generate"]
