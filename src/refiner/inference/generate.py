from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias

from refiner.inference._runtime import inference_map
from refiner.inference.client import InferenceResponse, _OpenAIEndpointClient
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult

GeneratePayload: TypeAlias = Mapping[str, Any]
GenerateFn: TypeAlias = Callable[[GeneratePayload], Awaitable[InferenceResponse]]
InferenceFn: TypeAlias = Callable[[Row, GenerateFn], Awaitable[MapResult] | MapResult]


def generate(
    *,
    fn: InferenceFn,
    provider: OpenAIEndpointProvider | VLLMProvider,
    default_generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    return inference_map(
        name="inference.generate",
        fn=fn,
        provider=provider,
        defaults=default_generation_params,
        defaults_key="default_generation_params",
        max_concurrent_requests=max_concurrent_requests,
        call=_generate,
        record=_record_usage,
    )


async def _generate(
    client: _OpenAIEndpointClient,
    payload: Mapping[str, Any],
) -> InferenceResponse:
    return await client.generate(payload)


def _record_usage(row: Row, response: Any) -> None:
    if not isinstance(response, InferenceResponse):
        return
    row.log_throughput(
        "prompt_tokens", _usage_int(response.usage, "prompt_tokens"), unit="tokens"
    )
    row.log_throughput(
        "completion_tokens",
        _usage_int(response.usage, "completion_tokens"),
        unit="tokens",
    )


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    raw = usage.get(key, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


__all__ = ["generate"]
