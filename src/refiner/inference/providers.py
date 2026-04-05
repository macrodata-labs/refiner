from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from refiner.services import RuntimeServiceSpec


@dataclass(frozen=True, slots=True)
class OpenAIEndpointProvider:
    base_url: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")

    def service_spec(self) -> RuntimeServiceSpec | None:
        return None

    def to_builtin_args(self) -> dict[str, object]:
        return {
            "type": "openai_endpoint",
            "base_url": self.base_url,
            "api_key": self.api_key,
        }


@dataclass(frozen=True, slots=True)
class VLLMProvider:
    model_name_or_path: str
    model_max_context: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")

    def service_spec(self) -> RuntimeServiceSpec:
        name_source = f"{self.model_name_or_path}\0{self.model_max_context}"
        name = f"vllm-{hashlib.sha1(name_source.encode('utf-8')).hexdigest()[:12]}"
        config: dict[str, Any] = {"model_name_or_path": self.model_name_or_path}
        if self.model_max_context is not None:
            config["model_max_context"] = self.model_max_context
        return RuntimeServiceSpec(name=name, kind="llm", config=config)

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "type": "vllm",
            "model_name_or_path": self.model_name_or_path,
        }
        if self.model_max_context is not None:
            payload["model_max_context"] = self.model_max_context
        return payload


__all__ = ["OpenAIEndpointProvider", "VLLMProvider"]
