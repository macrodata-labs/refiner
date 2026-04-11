from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from refiner.services.base import RuntimeServiceBinding, RuntimeServiceSpec


@dataclass(frozen=True, slots=True)
class VLLMServiceDefinition:
    model_name_or_path: str
    model_max_context: int | None = None
    kind: str = "llm"

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")

    @property
    def name(self) -> str:
        name_source = f"{self.model_name_or_path}\0{self.model_max_context}"
        return f"vllm-{hashlib.sha1(name_source.encode('utf-8')).hexdigest()[:12]}"

    def to_spec(self) -> RuntimeServiceSpec:
        config: dict[str, Any] = {"model_name_or_path": self.model_name_or_path}
        if self.model_max_context is not None:
            config["model_max_context"] = self.model_max_context
        return RuntimeServiceSpec(name=self.name, kind=self.kind, config=config)


@dataclass(frozen=True, slots=True)
class VLLMRuntimeServiceBinding(RuntimeServiceBinding):
    endpoint: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VLLMRuntimeServiceBinding:
        name = str(payload.get("name", "")).strip()
        kind = str(payload.get("kind", "")).strip()
        if not name:
            raise ValueError("service binding name must be non-empty")
        if not kind:
            raise ValueError("service binding kind must be non-empty")
        if kind != "llm":
            raise ValueError(f"unsupported VLLM runtime binding kind {kind!r}")
        endpoint = str(payload.get("endpoint", "")).strip()
        if not endpoint:
            raise ValueError(
                f"service binding {name!r} must include a non-empty endpoint"
            )
        return cls(
            name=name,
            kind=kind,
            endpoint=endpoint,
        )


__all__ = ["VLLMServiceDefinition", "VLLMRuntimeServiceBinding"]
