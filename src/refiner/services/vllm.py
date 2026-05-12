from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from refiner.services.base import RuntimeServiceBinding, RuntimeServiceSpec


@dataclass(frozen=True, slots=True)
class VLLMServiceDefinition:
    model_name_or_path: str
    config: Literal["correctness", "throughput"] = "correctness"
    kind: str = "llm"

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.config not in {"correctness", "throughput"}:
            raise ValueError("config must be 'correctness' or 'throughput'")

    @property
    def name(self) -> str:
        name_source = json.dumps(
            self._service_config(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"vllm-{hashlib.sha256(name_source.encode('utf-8')).hexdigest()[:12]}"

    def to_spec(self) -> RuntimeServiceSpec:
        return RuntimeServiceSpec(
            name=self.name, kind=self.kind, config=self._service_config()
        )

    def _service_config(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "config": self.config,
        }


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
