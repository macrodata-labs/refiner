from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from refiner.inference.internal.runtime import inference_map
from refiner.inference.providers import VLLMProvider
from refiner.inference.providers.openai import _OpenAIEndpointClient
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult

GeneratePoolingPayload: TypeAlias = Mapping[str, Any]
GeneratePoolingFn: TypeAlias = Callable[
    [GeneratePoolingPayload], Awaitable[Mapping[str, Any]]
]
GeneratePoolingMapFn: TypeAlias = Callable[
    [Row, GeneratePoolingFn], Awaitable[MapResult] | MapResult
]


def generate_pooling(
    *,
    fn: GeneratePoolingMapFn,
    provider: VLLMProvider,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    return inference_map(
        name="inference.generate_pooling",
        fn=fn,
        provider=provider,
        defaults=None,
        max_concurrent_requests=max_concurrent_requests,
        call=_generate_pooling,
    )


async def _generate_pooling(
    client: _OpenAIEndpointClient,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    return await client.pooling(payload)


__all__ = [
    "GeneratePoolingFn",
    "GeneratePoolingMapFn",
    "GeneratePoolingPayload",
    "generate_pooling",
]
