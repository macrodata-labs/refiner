from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from typing import Any

from loguru import logger

from refiner.services.base import RuntimeServiceBinding, RuntimeServiceSpec
from refiner.services.vllm import VLLMRuntimeServiceBinding

if TYPE_CHECKING:
    from refiner.platform.client.api import MacrodataClient

_POLL_INTERVAL_SECONDS = 2.0
_START_TIMEOUT_SECONDS = 20 * 60


class ServiceManager:
    def __init__(
        self,
        *,
        client: "MacrodataClient" | None = None,
        job_id: str | None = None,
        stage_index: int | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._client = client
        self._job_id = job_id
        self._stage_index = stage_index
        self._worker_id = worker_id
        self._started_by_name: dict[str, dict[str, str]] = {}
        self._resolved_by_name: dict[str, RuntimeServiceBinding] = {}
        self._pending_by_name: dict[str, asyncio.Task[RuntimeServiceBinding]] = {}

    async def start_services(
        self,
        services: Sequence[RuntimeServiceSpec],
    ) -> None:
        if services:
            for service in services:
                logger.info(
                    "Starting runtime service {}:{}",
                    service.kind,
                    service.name,
                )
        if self._client is None:
            raise RuntimeError(
                "Runtime service creation is only supported with cloud executor."
            )
        client = self._client
        worker_id = (self._worker_id or "").strip()
        if client is None or not worker_id:
            raise RuntimeError(
                "Cloud runtime service creation requires an active cloud worker context."
            )
        response = client.start_worker_services(
            job_id=self._job_id or "",
            stage_index=self._stage_index or 0,
            worker_id=worker_id,
            services=[service.to_dict() for service in services],
        )
        for item in _parse_started_services_response(response):
            self._add_runtime_binding(item)

    async def get(self, name: str) -> RuntimeServiceBinding:
        cached = self._resolved_by_name.get(name)
        if cached is not None:
            return cached

        started = self._started_by_name.get(name)
        if started is None:
            raise RuntimeError(f"service {name!r} was not started")

        pending = self._pending_by_name.get(name)
        if pending is not None:
            return await pending

        logger.info(
            "Waiting for runtime service {}:{}",
            started["kind"],
            started["name"],
        )
        task = asyncio.create_task(self._resolve_started_service(started))
        self._pending_by_name[name] = task
        try:
            binding = await task
        finally:
            self._pending_by_name.pop(name, None)
        self._resolved_by_name[name] = binding
        return binding

    def _add_runtime_binding(self, item: Mapping[str, Any]) -> None:
        name = str(item.get("name", "")).strip()
        kind = str(item.get("kind", "")).strip()
        if not name:
            raise ValueError("runtime binding name must be non-empty")
        if not kind:
            raise ValueError(f"runtime binding {name!r} kind must be non-empty")
        if name in self._started_by_name or name in self._resolved_by_name:
            raise ValueError(f"duplicate service name {name!r}")
        service_id = str(item.get("id", "")).strip()
        if not service_id:
            raise ValueError(f"runtime binding {name!r} must include an id")
        self._started_by_name[name] = {"id": service_id, "name": name, "kind": kind}

    async def _resolve_started_service(
        self, started: Mapping[str, str]
    ) -> RuntimeServiceBinding:
        client = self._client
        worker_id = (self._worker_id or "").strip()
        if client is None or not worker_id:
            raise RuntimeError(
                "Cloud runtime service resolution requires an active cloud worker context."
            )
        deadline = asyncio.get_running_loop().time() + _START_TIMEOUT_SECONDS
        while True:
            response = client.get_worker_service(
                job_id=self._job_id or "",
                stage_index=self._stage_index or 0,
                worker_id=worker_id,
                service_id=started["id"],
            )
            if not isinstance(response, Mapping):
                raise ValueError("runtime service response must be a JSON object")
            item = response.get("service", response)
            if not isinstance(item, Mapping):
                raise ValueError(
                    "runtime service status response must contain a service object"
                )
            status = str(item.get("status", "")).strip()
            if status == "failed":
                error = str(item.get("error", "")).strip() or "UnknownError"
                raise RuntimeError(
                    f"runtime service {started['name']!r} failed to start: {error}"
                )
            if status == "stopped":
                raise RuntimeError(
                    f"runtime service {started['name']!r} stopped before becoming ready"
                )
            if status == "ready":
                return _parse_runtime_service_binding(item, fallback=started)
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    "runtime service "
                    f"{started['name']!r} did not become ready within {_START_TIMEOUT_SECONDS} seconds"
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)


def _parse_runtime_service_binding(
    payload: Mapping[str, Any], *, fallback: Mapping[str, str]
) -> RuntimeServiceBinding:
    kind = str(payload.get("kind", "")).strip() or fallback["kind"]
    normalized = dict(payload)
    normalized["name"] = str(payload.get("name", "")).strip() or fallback["name"]
    normalized["kind"] = kind
    if kind == "llm":
        return VLLMRuntimeServiceBinding.from_dict(normalized)
    raise ValueError(f"unsupported service binding kind {kind!r}")


def _parse_started_services_response(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    services = payload.get("services")
    if not isinstance(services, Sequence):
        raise ValueError("runtime bindings response must contain a services list")
    for item in services:
        if not isinstance(item, Mapping):
            raise ValueError("runtime bindings entries must be objects")
    return tuple(services)


__all__ = ["ServiceManager"]
