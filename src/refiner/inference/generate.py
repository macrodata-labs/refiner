from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, cast

from refiner.inference.client import InferenceResponse, _OpenAIEndpointClient
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_manager
from refiner.worker.metrics.api import register_gauge

_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"

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
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    client: _OpenAIEndpointClient | None = None
    client_lock = asyncio.Lock()
    request_semaphore = asyncio.Semaphore(max_concurrent_requests)
    gauges_registered = False
    waiting_requests = 0
    running_requests = 0

    def _ensure_metrics_registered() -> None:
        nonlocal gauges_registered
        if gauges_registered:
            return
        register_gauge(
            "waiting_requests",
            lambda: waiting_requests,
            unit="requests",
        )
        register_gauge(
            "running_requests",
            lambda: running_requests,
            unit="requests",
        )
        gauges_registered = True

    async def _generate(row: Row, payload: GeneratePayload) -> InferenceResponse:
        nonlocal client
        nonlocal running_requests
        nonlocal waiting_requests
        request_payload: dict[str, Any] = {}
        if isinstance(provider, OpenAIEndpointProvider):
            request_payload["model"] = provider.model
        request_payload.update(dict(default_generation_params or {}))
        request_payload.update(dict(payload))
        if client is None:
            async with client_lock:
                if client is None:
                    if isinstance(provider, OpenAIEndpointProvider):
                        client = _OpenAIEndpointClient(
                            base_url=provider.base_url,
                        )
                    else:
                        service_name = provider.service_definition().name
                        service_manager = get_active_service_manager()
                        binding = None
                        if service_manager is not None:
                            binding = await service_manager.get(service_name)
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
                    _ensure_metrics_registered()
            assert client is not None

        waiting_requests += 1
        await request_semaphore.acquire()
        waiting_requests -= 1
        running_requests += 1
        try:
            response = await client.generate(request_payload)
        except Exception:
            row.log_throughput("failed_requests", 1, unit="requests")
            raise
        finally:
            running_requests -= 1
            request_semaphore.release()
        row.log_throughput("successful_requests", 1, unit="requests")
        prompt_tokens = _usage_int(response.usage, "prompt_tokens")
        completion_tokens = _usage_int(response.usage, "completion_tokens")
        row.log_throughput("prompt_tokens", prompt_tokens, unit="tokens")
        row.log_throughput("completion_tokens", completion_tokens, unit="tokens")
        return response

    async def _wrapped(row: Row) -> MapResult:
        result = fn(row, lambda payload: _generate(row, payload))
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    setattr(
        _wrapped,
        _REFINER_BUILTIN_CALL_ATTR,
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


def _usage_int(usage: Mapping[str, Any], key: str) -> int:
    raw = usage.get(key, 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0
