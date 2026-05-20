from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, cast

from refiner.inference.client import _OpenAIEndpointClient
from refiner.inference.providers import VLLMProvider
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_manager

_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"

PoolingPayload: TypeAlias = Mapping[str, Any]
PoolingFn: TypeAlias = Callable[[PoolingPayload], Awaitable[Mapping[str, Any]]]
PoolingMapFn: TypeAlias = Callable[[Row, PoolingFn], Awaitable[MapResult] | MapResult]


def pooling(
    *,
    fn: PoolingMapFn,
    provider: VLLMProvider,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    client: _OpenAIEndpointClient | None = None
    client_lock = asyncio.Lock()
    request_semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _request(payload: PoolingPayload) -> Mapping[str, Any]:
        nonlocal client
        request_payload = {"model": provider.model, **dict(payload)}
        if client is None:
            async with client_lock:
                if client is None:
                    service_name = provider.service_definition().name
                    service_manager = get_active_service_manager()
                    binding = (
                        None
                        if service_manager is None
                        else await service_manager.get(service_name)
                    )
                    if not isinstance(binding, VLLMRuntimeServiceBinding):
                        raise RuntimeError(
                            f"VLLM provider requires runtime service binding {service_name!r}"
                        )
                    client = _OpenAIEndpointClient(
                        base_url=binding.endpoint,
                        api_key=binding.api_key,
                    )

        async with request_semaphore:
            assert client is not None
            return await client.pooling(request_payload)

    async def _wrapped(row: Row) -> MapResult:
        result = fn(row, _request)
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    service = provider.service_definition()
    setattr(
        _wrapped,
        _REFINER_BUILTIN_CALL_ATTR,
        {
            "name": "inference.pooling",
            "args": {
                "fn": fn,
                "provider": provider.to_builtin_args(),
                "max_concurrent_requests": max_concurrent_requests,
            },
            "services": [service.to_spec()],
        },
    )
    return _wrapped


__all__ = ["pooling", "PoolingFn", "PoolingMapFn", "PoolingPayload"]
