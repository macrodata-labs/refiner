from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from refiner.inference.client import PoolingResponse, _OpenAIEndpointClient
from refiner.inference._runtime import build_inference_map
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult

PoolingPayload: TypeAlias = Mapping[str, Any]
PoolingFn: TypeAlias = Callable[[PoolingPayload], Awaitable[PoolingResponse]]
PoolingMapFn: TypeAlias = Callable[
    [Row, PoolingFn],
    Awaitable[MapResult] | MapResult,
]


def pooling(
    *,
    fn: PoolingMapFn,
    provider: OpenAIEndpointProvider | VLLMProvider,
    default_pooling_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    return build_inference_map(
        name="inference.pooling",
        fn=fn,
        provider=provider,
        default_params=default_pooling_params,
        default_params_arg="default_pooling_params",
        max_concurrent_requests=max_concurrent_requests,
        client_call=_pooling,
        metrics_prefix="pooling",
    )


__all__ = ["pooling", "PoolingFn", "PoolingMapFn", "PoolingPayload"]


async def _pooling(
    client: _OpenAIEndpointClient,
    payload: Mapping[str, Any],
) -> PoolingResponse:
    return await client.pooling(payload)
