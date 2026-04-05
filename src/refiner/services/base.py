from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeServiceSpec:
    name: str
    kind: str
    config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "config": dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class RuntimeServiceBinding:
    name: str
    kind: str


def parse_runtime_service_specs(
    payload: Mapping[str, Any] | None,
) -> tuple[RuntimeServiceSpec, ...]:
    if payload is None:
        return ()
    services = payload.get("services")
    if services is None:
        return ()
    if not isinstance(services, Sequence):
        raise ValueError("runtime services payload must contain a services list")
    parsed: list[RuntimeServiceSpec] = []
    seen_names: set[str] = set()
    for item in services:
        if not isinstance(item, Mapping):
            raise ValueError("runtime services entries must be objects")
        name = str(item.get("name", "")).strip()
        kind = str(item.get("kind", "")).strip()
        config = item.get("config")
        if not name:
            raise ValueError("runtime service name must be non-empty")
        if not kind:
            raise ValueError(f"runtime service {name!r} kind must be non-empty")
        if not isinstance(config, Mapping):
            raise ValueError(f"runtime service {name!r} config must be an object")
        if name in seen_names:
            raise ValueError(f"duplicate runtime service name {name!r}")
        seen_names.add(name)
        parsed.append(RuntimeServiceSpec(name=name, kind=kind, config=dict(config)))
    return tuple(parsed)


def parse_runtime_service_bindings(
    payload: Mapping[str, Any] | None,
) -> tuple[RuntimeServiceBinding, ...]:
    if payload is None:
        return ()
    services = payload.get("services")
    if services is None:
        return ()
    if not isinstance(services, Sequence):
        raise ValueError("service bindings payload must contain a services list")
    parsed: list[RuntimeServiceBinding] = []
    seen_names: set[str] = set()
    for item in services:
        if not isinstance(item, Mapping):
            raise ValueError("service bindings entries must be objects")
        kind = str(item.get("kind", "")).strip()
        from refiner.services.vllm import VLLMRuntimeServiceBinding

        if kind == "llm":
            binding = VLLMRuntimeServiceBinding.from_dict(item)
        else:
            raise ValueError(f"unsupported service binding kind {kind!r}")
        if binding.name in seen_names:
            raise ValueError(f"duplicate service binding name {binding.name!r}")
        seen_names.add(binding.name)
        parsed.append(binding)
    return tuple(parsed)


__all__ = [
    "RuntimeServiceBinding",
    "RuntimeServiceSpec",
    "parse_runtime_service_specs",
    "parse_runtime_service_bindings",
]
