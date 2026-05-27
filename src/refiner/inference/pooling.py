from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from refiner.inference.internal.runtime import inference_map
from refiner.inference.providers import VLLMProvider
from refiner.inference.providers.openai import _OpenAIEndpointClient
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult

PoolingPayload: TypeAlias = Mapping[str, Any]
PoolingFn: TypeAlias = Callable[[PoolingPayload], Awaitable[Mapping[str, Any]]]
PoolingMapFn: TypeAlias = Callable[[Row, PoolingFn], Awaitable[MapResult] | MapResult]


def pooling(
    *,
    fn: PoolingMapFn,
    provider: VLLMProvider,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    return inference_map(
        name="inference.pooling",
        fn=fn,
        provider=provider,
        defaults=None,
        max_concurrent_requests=max_concurrent_requests,
        call=_pooling,
    )


async def _pooling(
    client: _OpenAIEndpointClient,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    return await client.pooling(payload)


__all__ = ["pooling", "PoolingFn", "PoolingMapFn", "PoolingPayload"]
