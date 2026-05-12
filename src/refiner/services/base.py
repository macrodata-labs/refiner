from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeServiceSpec:
    name: str
    kind: str
    config: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeServiceSpec:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("runtime service name must be non-empty")
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"runtime service {name!r} kind must be non-empty")
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"runtime service {name!r} config must be an object")
        return cls(name=name.strip(), kind=kind.strip(), config=dict(config))

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


__all__ = [
    "RuntimeServiceBinding",
    "RuntimeServiceSpec",
]
