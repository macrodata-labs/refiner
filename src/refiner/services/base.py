from __future__ import annotations

from collections.abc import Mapping
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


__all__ = [
    "RuntimeServiceBinding",
    "RuntimeServiceSpec",
]
