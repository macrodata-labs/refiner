from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, cast

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
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_manager
from refiner.worker.metrics.api import register_gauge

_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"

Provider: TypeAlias = (
    AnthropicEndpointProvider
    | GoogleEndpointProvider
    | OpenAIEndpointProvider
    | OpenAIResponsesProvider
    | VLLMProvider
)
RequestFn: TypeAlias = Callable[[Mapping[str, Any]], Awaitable[Any]]
MapFn: TypeAlias = Callable[[Row, RequestFn], Awaitable[MapResult] | MapResult]
ClientCall: TypeAlias = Callable[[Any, Mapping[str, Any]], Awaitable[Any]]


def inference_map(
    *,
    name: str,
    fn: MapFn,
    provider: Provider,
    defaults: Mapping[str, Any] | None,
    defaults_key: str | None = None,
    merge_defaults: bool = True,
    max_concurrent_requests: int = 256,
    call: ClientCall,
    record: Callable[[Row, Any], None] | None = None,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    client: (
        _AnthropicEndpointClient
        | _GoogleEndpointClient
        | _OpenAIEndpointClient
        | _OpenAIResponsesClient
        | None
    ) = None
    client_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    gauges_registered = False
    waiting_requests = 0
    running_requests = 0

    def _register_metrics() -> None:
        nonlocal gauges_registered
        if gauges_registered:
            return
        register_gauge("waiting_requests", lambda: waiting_requests, unit="requests")
        register_gauge("running_requests", lambda: running_requests, unit="requests")
        gauges_registered = True

    async def _client() -> (
        _AnthropicEndpointClient
        | _GoogleEndpointClient
        | _OpenAIEndpointClient
        | _OpenAIResponsesClient
    ):
        nonlocal client
        if client is not None:
            return client
        async with client_lock:
            if client is not None:
                return client
            if isinstance(provider, OpenAIEndpointProvider):
                client = _OpenAIEndpointClient(
                    base_url=provider.base_url,
                )
            elif isinstance(provider, OpenAIResponsesProvider):
                client = _OpenAIResponsesClient(
                    base_url=provider.base_url,
                )
            elif isinstance(provider, GoogleEndpointProvider):
                client = _GoogleEndpointClient(
                    base_url=provider.base_url,
                    model=provider.model,
                )
            elif isinstance(provider, AnthropicEndpointProvider):
                client = _AnthropicEndpointClient(
                    base_url=provider.base_url,
                    anthropic_version=provider.anthropic_version,
                )
            else:
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
            return client

    async def _request(row: Row, payload: Mapping[str, Any]) -> Any:
        nonlocal running_requests, waiting_requests
        request_payload = {
            **(dict(defaults or {}) if merge_defaults else {}),
            **dict(payload),
        }
        if not isinstance(provider, GoogleEndpointProvider):
            request_payload = {"model": provider.model, **request_payload}
        _register_metrics()
        waiting_requests += 1
        await semaphore.acquire()
        waiting_requests -= 1
        request_running = False
        try:
            resolved_client = await _client()
            running_requests += 1
            request_running = True
            response = await call(resolved_client, request_payload)
        except Exception:
            row.log_throughput("failed_requests", 1, unit="requests")
            raise
        finally:
            if request_running:
                running_requests -= 1
            semaphore.release()
        row.log_throughput("successful_requests", 1, unit="requests")
        if record is not None:
            record(row, response)
        return response

    async def _wrapped(row: Row) -> MapResult:
        result = fn(row, lambda payload: _request(row, payload))
        if inspect.isawaitable(result):
            return cast(MapResult, await result)
        return cast(MapResult, result)

    service = provider.service_definition()
    args: dict[str, Any] = {
        "fn": fn,
        "provider": provider.to_builtin_args(),
        "max_concurrent_requests": max_concurrent_requests,
    }
    if defaults_key is not None:
        args[defaults_key] = dict(defaults or {})
    setattr(
        _wrapped,
        _REFINER_BUILTIN_CALL_ATTR,
        {
            "name": name,
            "args": args,
            "services": [] if service is None else [service.to_spec()],
        },
    )
    return _wrapped


__all__ = ["_REFINER_BUILTIN_CALL_ATTR", "inference_map"]
