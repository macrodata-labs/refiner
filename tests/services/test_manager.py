from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest

from refiner.services import (
    ServiceManager,
    VLLMRuntimeServiceBinding,
    VLLMServiceDefinition,
)
from refiner.services import manager as manager_module


class _FakeClient:
    def __init__(self, *, start_response, get_service):
        self._start_response = start_response
        self._get_service = get_service

    def start_worker_services(self, **kwargs):
        del kwargs["job_id"], kwargs["stage_index"], kwargs["worker_id"]
        services = kwargs["services"]
        return (
            self._start_response(services)
            if callable(self._start_response)
            else self._start_response
        )

    def get_worker_service(self, **kwargs):
        del kwargs["job_id"], kwargs["stage_index"], kwargs["worker_id"]
        service_id = kwargs["service_id"]
        return self._get_service(service_id)


def test_service_manager_rejects_unknown_service() -> None:
    client = _FakeClient(
        start_response={
            "services": [
                {"id": "svc-1", "name": "llm-a", "kind": "llm"},
            ]
        },
        get_service=lambda service_id: {
            "service": {"id": service_id, "status": "ready"}
        },
    )
    manager = ServiceManager(
        client=cast(Any, client),
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
    )
    asyncio.run(
        manager.start_services(
            [
                VLLMServiceDefinition(
                    model_name_or_path="meta-llama/Llama-3.1-8B"
                ).to_spec()
            ]
        )
    )

    with pytest.raises(RuntimeError, match="service 'missing' was not started"):
        asyncio.run(manager.get("missing"))


def test_service_manager_rejects_missing_client_startup() -> None:
    manager = ServiceManager()

    with pytest.raises(
        RuntimeError,
        match="Runtime service creation is only supported with cloud executor.",
    ):
        asyncio.run(
            manager.start_services(
                [
                    VLLMServiceDefinition(
                        model_name_or_path="meta-llama/Llama-3.1-8B"
                    ).to_spec()
                ]
            )
        )


def test_service_manager_polls_until_service_is_ready() -> None:
    seen: dict[str, Any] = {"calls": 0}

    def _get_service(service_id: str) -> Mapping[str, object]:
        seen["service_id"] = service_id
        seen["calls"] += 1
        if seen["calls"] == 1:
            return {
                "service": {
                    "id": service_id,
                    "name": "llm-a",
                    "kind": "llm",
                    "status": "starting",
                }
            }
        return {
            "service": {
                "id": service_id,
                "name": "llm-a",
                "kind": "llm",
                "status": "ready",
                "endpoint": "http://127.0.0.1:9000",
                "api_key": "service-secret",
            }
        }

    client = _FakeClient(
        start_response={
            "services": [
                {"id": "svc-1", "name": "llm-a", "kind": "llm"},
            ]
        },
        get_service=_get_service,
    )
    manager = ServiceManager(
        client=cast(Any, client),
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
    )
    asyncio.run(
        manager.start_services(
            [
                VLLMServiceDefinition(
                    model_name_or_path="meta-llama/Llama-3.1-8B"
                ).to_spec()
            ]
        )
    )
    binding = asyncio.run(manager.get("llm-a"))

    assert binding == VLLMRuntimeServiceBinding(
        name="llm-a",
        kind="llm",
        endpoint="http://127.0.0.1:9000",
        api_key="service-secret",
    )
    assert seen["service_id"] == "svc-1"
    assert seen["calls"] == 2


def test_service_manager_caches_resolved_service_binding() -> None:
    seen: dict[str, int] = {"calls": 0}

    def _get_service(service_id: str) -> Mapping[str, object]:
        seen["calls"] += 1
        return {
            "service": {
                "id": service_id,
                "name": "llm-a",
                "kind": "llm",
                "status": "ready",
                "endpoint": "http://127.0.0.1:9000",
                "api_key": "service-secret",
            }
        }

    client = _FakeClient(
        start_response={
            "services": [
                {"id": "svc-1", "name": "llm-a", "kind": "llm"},
            ]
        },
        get_service=_get_service,
    )
    manager = ServiceManager(
        client=cast(Any, client),
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
    )
    asyncio.run(
        manager.start_services(
            [
                VLLMServiceDefinition(
                    model_name_or_path="meta-llama/Llama-3.1-8B"
                ).to_spec()
            ]
        )
    )
    first = asyncio.run(manager.get("llm-a"))
    second = asyncio.run(manager.get("llm-a"))

    assert first is second
    assert seen["calls"] == 1


def test_service_manager_times_out_when_service_never_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _get_service(service_id: str) -> Mapping[str, object]:
        return {
            "service": {
                "id": service_id,
                "name": "llm-a",
                "kind": "llm",
                "status": "starting",
            }
        }

    monkeypatch.setattr(manager_module, "_START_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(manager_module, "_POLL_INTERVAL_SECONDS", 0.001)

    client = _FakeClient(
        start_response={
            "services": [
                {"id": "svc-1", "name": "llm-a", "kind": "llm"},
            ]
        },
        get_service=_get_service,
    )
    manager = ServiceManager(
        client=cast(Any, client),
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
    )
    asyncio.run(
        manager.start_services(
            [
                VLLMServiceDefinition(
                    model_name_or_path="meta-llama/Llama-3.1-8B"
                ).to_spec()
            ]
        )
    )
    with pytest.raises(
        RuntimeError,
        match="runtime service 'llm-a' did not become ready within 0.01 seconds",
    ):
        asyncio.run(manager.get("llm-a"))


def test_service_manager_can_start_requested_services_lazily() -> None:
    seen: dict[str, Any] = {"start_calls": 0, "status_calls": 0}
    definition = VLLMServiceDefinition(model_name_or_path="meta-llama/Llama-3.1-8B")

    def _start_services(services):
        seen["start_calls"] += 1
        assert services == [definition.to_spec().to_dict()]
        return {
            "services": [
                {
                    "id": "svc-1",
                    "name": definition.name,
                    "kind": "llm",
                }
            ]
        }

    def _get_service(service_id: str) -> Mapping[str, object]:
        seen["status_calls"] += 1
        return {
            "service": {
                "id": service_id,
                "name": definition.name,
                "kind": "llm",
                "status": "ready",
                "endpoint": "http://127.0.0.1:9000",
                "api_key": "service-secret",
            }
        }

    client = _FakeClient(start_response=_start_services, get_service=_get_service)
    manager = ServiceManager(
        client=cast(Any, client),
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
    )
    asyncio.run(manager.start_services([definition.to_spec()]))
    binding = asyncio.run(manager.get(definition.name))

    assert binding == VLLMRuntimeServiceBinding(
        name=definition.name,
        kind="llm",
        endpoint="http://127.0.0.1:9000",
        api_key="service-secret",
    )
    assert seen["start_calls"] == 1
    assert seen["status_calls"] == 1
