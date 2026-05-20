from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from refiner.inference.client import _OpenAIEndpointClient
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_manager
from refiner.worker.metrics.api import register_gauge

_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"

Provider = OpenAIEndpointProvider | VLLMProvider
RequestFn = Callable[[Mapping[str, Any]], Awaitable[Any]]
InferenceMapFn = Callable[[Row, RequestFn], Awaitable[MapResult] | MapResult]
ClientCall = Callable[[_OpenAIEndpointClient, Mapping[str, Any]], Awaitable[Any]]
ResponseHook = Callable[[Row, Any], None]


def build_inference_map(
    *,
    name: str,
    fn: InferenceMapFn,
    provider: Provider,
    default_params: Mapping[str, Any] | None,
    default_params_arg: str,
    max_concurrent_requests: int,
    client_call: ClientCall,
    metrics_prefix: str = "",
    response_hook: ResponseHook | None = None,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    client: _OpenAIEndpointClient | None = None
    client_lock = asyncio.Lock()
    request_semaphore = asyncio.Semaphore(max_concurrent_requests)
    gauges_registered = False
    waiting_requests = 0
    running_requests = 0

    def metric(label: str) -> str:
        return f"{metrics_prefix}_{label}" if metrics_prefix else label

    def _ensure_metrics_registered() -> None:
        nonlocal gauges_registered
        if gauges_registered:
            return
        register_gauge(
            metric("waiting_requests"), lambda: waiting_requests, unit="requests"
        )
        register_gauge(
            metric("running_requests"), lambda: running_requests, unit="requests"
        )
        gauges_registered = True

    async def _request(row: Row, payload: Mapping[str, Any]) -> Any:
        nonlocal client
        nonlocal running_requests
        nonlocal waiting_requests
        request_payload = {"model": provider.model}
        request_payload.update(dict(default_params or {}))
        request_payload.update(dict(payload))
        if client is None:
            async with client_lock:
                if client is None:
                    client = await _create_client(provider)
                    _ensure_metrics_registered()

        waiting_requests += 1
        await request_semaphore.acquire()
        waiting_requests -= 1
        running_requests += 1
        try:
            response = await client_call(client, request_payload)
        except Exception:
            row.log_throughput(metric("failed_requests"), 1, unit="requests")
            raise
        finally:
            running_requests -= 1
            request_semaphore.release()
        row.log_throughput(metric("successful_requests"), 1, unit="requests")
        if response_hook is not None:
            response_hook(row, response)
        return response

    async def _wrapped(row: Row) -> MapResult:
        result = fn(row, lambda payload: _request(row, payload))
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    setattr(
        _wrapped,
        _REFINER_BUILTIN_CALL_ATTR,
        {
            "name": name,
            "args": {
                "fn": fn,
                "provider": provider.to_builtin_args(),
                default_params_arg: dict(default_params or {}),
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


async def _create_client(provider: Provider) -> _OpenAIEndpointClient:
    if isinstance(provider, OpenAIEndpointProvider):
        return _OpenAIEndpointClient(base_url=provider.base_url)
    service_name = provider.service_definition().name
    service_manager = get_active_service_manager()
    binding = (
        None if service_manager is None else await service_manager.get(service_name)
    )
    if binding is None:
        raise RuntimeError(
            f"VLLM provider requires runtime service binding {service_name!r}"
        )
    if not isinstance(binding, VLLMRuntimeServiceBinding):
        raise RuntimeError(
            f"VLLM provider expected a VLLM runtime binding for {service_name!r}"
        )
    return _OpenAIEndpointClient(base_url=binding.endpoint, api_key=binding.api_key)


__all__ = [
    "_REFINER_BUILTIN_CALL_ATTR",
    "InferenceMapFn",
    "Provider",
    "RequestFn",
    "build_inference_map",
]
