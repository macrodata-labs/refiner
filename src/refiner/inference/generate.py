from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.services import BaseGenerationService, RuntimeServiceDefinition
from refiner.worker.context import get_active_service_registry

GeneratePayload: TypeAlias = Mapping[str, Any]
InferenceFn: TypeAlias = Callable[
    [Row, BaseGenerationService],
    Awaitable[MapResult] | MapResult,
]

_BUILTIN_ATTR = "__refiner_builtin_call__"
_ASYNC_STEP_MAX_IN_FLIGHT_ATTR = "__refiner_async_step_max_in_flight__"
_ASYNC_STEP_PRESERVE_ORDER_ATTR = "__refiner_async_step_preserve_order__"


@dataclass(frozen=True, slots=True)
class _GenerateConfig:
    service_name: str
    default_generation_params: Mapping[str, Any] | None = None
    max_in_flight: int = 128
    max_concurrent_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.service_name.strip():
            raise ValueError("service_name must be non-empty")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")
        if self.max_concurrent_rows is not None and self.max_concurrent_rows <= 0:
            raise ValueError("max_concurrent_rows must be > 0 when provided")

    @property
    def row_limit(self) -> int:
        return self.max_concurrent_rows or self.max_in_flight


@dataclass(slots=True)
class _ManagedGenerationService(BaseGenerationService):
    inner: BaseGenerationService
    default_generation_params: Mapping[str, Any]
    request_semaphore: asyncio.Semaphore

    async def generate(self, payload: Mapping[str, Any]) -> Any:
        request_payload = dict(self.default_generation_params)
        request_payload.update(dict(payload))
        async with self.request_semaphore:
            return await self.inner.generate(request_payload)


class _GenerateWrapper:
    def __init__(self, fn: InferenceFn, config: _GenerateConfig):
        self.fn = fn
        self.config = config
        self._service_definition: RuntimeServiceDefinition | None = None
        self._direct_client: BaseGenerationService | None = None
        self._direct_client_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(config.max_in_flight)
        self._row_semaphore = asyncio.Semaphore(config.row_limit)
        setattr(
            self,
            _BUILTIN_ATTR,
            {
                "name": "inference.generate",
                "args": {
                    "fn": fn,
                    "service": config.service_name,
                    "default_generation_params": dict(
                        config.default_generation_params or {}
                    ),
                    "max_in_flight": config.max_in_flight,
                    "max_concurrent_rows": config.max_concurrent_rows,
                },
            },
        )
        setattr(self, _ASYNC_STEP_MAX_IN_FLIGHT_ATTR, config.row_limit)
        setattr(self, _ASYNC_STEP_PRESERVE_ORDER_ATTR, True)

    async def __call__(self, row: Row) -> MapResult:
        service = await self._resolve_service()
        managed = _ManagedGenerationService(
            inner=service,
            default_generation_params=dict(self.config.default_generation_params or {}),
            request_semaphore=self._request_semaphore,
        )
        async with self._row_semaphore:
            result = self.fn(row, managed)
            if inspect.isawaitable(result):
                result = await cast(Awaitable[MapResult], result)
            return cast(MapResult, result)

    async def _resolve_service(self) -> BaseGenerationService:
        registry = get_active_service_registry()
        if registry is not None:
            return registry.get(self.config.service_name)
        if self._direct_client is not None:
            return self._direct_client
        async with self._direct_client_lock:
            if self._direct_client is None:
                if self._service_definition is None:
                    raise ValueError(
                        f"inference.generate(service_name={self.config.service_name!r}) "
                        "requires the service to be declared on map_async(..., services=[...])"
                    )
                self._direct_client = self._service_definition.build_client(None)
            return self._direct_client

    def bind_runtime_services(
        self, services: tuple[RuntimeServiceDefinition, ...]
    ) -> None:
        match: RuntimeServiceDefinition | None = None
        for service in services:
            if service.name != self.config.service_name:
                continue
            if match is not None and match != service:
                raise ValueError(
                    f"duplicate service definition name {service.name!r} with mismatched configuration"
                )
            match = service
        if match is None:
            raise ValueError(
                f"map_async(..., services=[...]) is missing service {self.config.service_name!r}"
            )
        self._service_definition = match


def generate(
    *,
    service_name: str,
    fn: InferenceFn,
    default_generation_params: Mapping[str, Any] | None = None,
    max_in_flight: int = 128,
    max_concurrent_rows: int | None = None,
) -> Callable[[Row], Awaitable[MapResult]]:
    config = _GenerateConfig(
        service_name=service_name,
        default_generation_params=default_generation_params,
        max_in_flight=max_in_flight,
        max_concurrent_rows=max_concurrent_rows,
    )
    return _GenerateWrapper(fn=fn, config=config)


__all__ = ["generate"]
