from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Any, TypeAlias, cast

from refiner.inference.client import GenerationRateLimitError, _OpenAIEndpointClient
from refiner.inference.providers import OpenAIEndpointProvider, VLLMProvider
from refiner.inference.rate_limit import (
    AdaptiveRateLimit,
    AdaptiveRateLimiter,
    RateLimit,
    StaticRateLimit,
)
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import VLLMRuntimeServiceBinding
from refiner.worker.context import get_active_service_manager
from refiner.worker.metrics.api import register_gauge

_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"

Provider: TypeAlias = OpenAIEndpointProvider | VLLMProvider
RequestFn: TypeAlias = Callable[[Mapping[str, Any]], Awaitable[Any]]
MapFn: TypeAlias = Callable[[Row, RequestFn], Awaitable[MapResult] | MapResult]
ClientCall: TypeAlias = Callable[
    [_OpenAIEndpointClient, Mapping[str, Any]], Awaitable[Any]
]


def inference_map(
    *,
    name: str,
    fn: MapFn,
    provider: Provider,
    defaults: Mapping[str, Any] | None,
    defaults_key: str | None = None,
    max_concurrent_requests: int = 256,
    rate_limit: RateLimit | None = None,
    rate_limit_key: str | None = None,
    call: ClientCall,
    record: Callable[[Row, Any], None] | None = None,
) -> Callable[[Row], Awaitable[MapResult]]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")
    resolved_rate_limit = rate_limit or StaticRateLimit(
        max_concurrency=max_concurrent_requests
    )
    client: _OpenAIEndpointClient | None = None
    client_lock = asyncio.Lock()
    adaptive_limiter: AdaptiveRateLimiter | None = (
        AdaptiveRateLimiter(resolved_rate_limit)
        if isinstance(resolved_rate_limit, AdaptiveRateLimit)
        else None
    )
    semaphore = (
        None
        if adaptive_limiter is not None
        else asyncio.Semaphore(resolved_rate_limit.max_concurrency)
    )
    gauges_registered = False
    waiting_requests = 0
    running_requests = 0

    def _register_metrics() -> None:
        nonlocal gauges_registered
        if gauges_registered:
            return
        register_gauge("waiting_requests", lambda: waiting_requests, unit="requests")
        register_gauge("running_requests", lambda: running_requests, unit="requests")
        if adaptive_limiter is not None:
            register_gauge(
                "adaptive_concurrency",
                lambda: adaptive_limiter.limit,
                unit="requests",
            )
        gauges_registered = True

    async def _client() -> _OpenAIEndpointClient:
        nonlocal client
        if client is not None:
            return client
        async with client_lock:
            if client is not None:
                return client
            if isinstance(provider, OpenAIEndpointProvider):
                client = _OpenAIEndpointClient(base_url=provider.base_url)
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
            _register_metrics()
            return client

    async def _request(row: Row, payload: Mapping[str, Any]) -> Any:
        nonlocal running_requests, waiting_requests
        request_payload = {
            "model": provider.model,
            **dict(defaults or {}),
            **dict(payload),
        }
        resolved_client = await _client()
        waiting_requests += 1
        acquired_adaptive = False
        try:
            if adaptive_limiter is not None:
                await adaptive_limiter.acquire()
                acquired_adaptive = True
            else:
                assert semaphore is not None
                await semaphore.acquire()
        finally:
            waiting_requests -= 1
        running_requests += 1
        try:
            response = await call(resolved_client, request_payload)
        except GenerationRateLimitError as err:
            if adaptive_limiter is not None:
                await adaptive_limiter.record_rate_limit(err.retry_after_seconds)
            row.log_throughput("rate_limited_requests", 1, unit="requests")
            row.log_throughput("failed_requests", 1, unit="requests")
            raise
        except Exception:
            row.log_throughput("failed_requests", 1, unit="requests")
            raise
        finally:
            running_requests -= 1
            if acquired_adaptive:
                assert adaptive_limiter is not None
                await adaptive_limiter.release()
            if semaphore is not None:
                semaphore.release()
        if adaptive_limiter is not None:
            await adaptive_limiter.record_success()
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
    }
    if rate_limit_key is None:
        args["max_concurrent_requests"] = resolved_rate_limit.max_concurrency
    else:
        args[rate_limit_key] = {
            "type": "adaptive"
            if isinstance(resolved_rate_limit, AdaptiveRateLimit)
            else "static",
            **asdict(resolved_rate_limit),
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
