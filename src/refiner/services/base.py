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
    endpoint: str
    headers: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeServiceBinding:
        name = str(payload.get("name", "")).strip()
        kind = str(payload.get("kind", "")).strip()
        endpoint = str(payload.get("endpoint", "")).strip()
        if not name:
            raise ValueError("service binding name must be non-empty")
        if not kind:
            raise ValueError("service binding kind must be non-empty")
        if not endpoint:
            raise ValueError(
                f"service binding {name!r} must include a non-empty endpoint"
            )
        headers = payload.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ValueError(f"service binding {name!r} headers must be an object")
            normalized_headers = {
                str(key): str(value) for key, value in headers.items()
            }
        else:
            normalized_headers = None
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"service binding {name!r} metadata must be an object")
        return cls(
            name=name,
            kind=kind,
            endpoint=endpoint,
            headers=normalized_headers,
            metadata=metadata,
        )


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
        binding = RuntimeServiceBinding.from_dict(item)
        if binding.name in seen_names:
            raise ValueError(f"duplicate service binding name {binding.name!r}")
        seen_names.add(binding.name)
        parsed.append(binding)
    return tuple(parsed)


__all__ = [
    "RuntimeServiceBinding",
    "RuntimeServiceSpec",
    "parse_runtime_service_bindings",
]
