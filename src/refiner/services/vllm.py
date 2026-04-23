from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from refiner.services.base import RuntimeServiceBinding, RuntimeServiceSpec


@dataclass(frozen=True, slots=True)
class VLLMServiceDefinition:
    model_name_or_path: str
    model_max_context: int | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "llm"

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")
        for key, value in dict(self.extra_kwargs).items():
            if not str(key).strip():
                raise ValueError("extra_kwargs keys must be non-empty")
            if value is None:
                raise ValueError("extra_kwargs values must be non-null")

    @property
    def name(self) -> str:
        name_source = json.dumps(
            {
                "model_name_or_path": self.model_name_or_path,
                "model_max_context": self.model_max_context,
                "extra_kwargs": self.extra_kwargs,
            },
            separators=(",", ":"),
        )
        return f"vllm-{hashlib.sha1(name_source.encode('utf-8')).hexdigest()[:12]}"

    def to_spec(self) -> RuntimeServiceSpec:
        config: dict[str, Any] = {"model_name_or_path": self.model_name_or_path}
        if self.model_max_context is not None:
            config["model_max_context"] = self.model_max_context
        if self.extra_kwargs:
            for key, value in self.extra_kwargs.items():
                config[str(key)] = value
        return RuntimeServiceSpec(name=self.name, kind=self.kind, config=config)


@dataclass(frozen=True, slots=True)
class VLLMRuntimeServiceBinding(RuntimeServiceBinding):
    endpoint: str
    api_key: str | None = None

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
        api_key = (
            None
            if payload.get("api_key") is None
            else str(payload.get("api_key")).strip() or None
        )
        return cls(
            name=name,
            kind=kind,
            endpoint=endpoint,
            api_key=api_key,
        )


__all__ = ["VLLMServiceDefinition", "VLLMRuntimeServiceBinding"]
