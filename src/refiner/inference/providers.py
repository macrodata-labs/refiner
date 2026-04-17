from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from refiner.services import VLLMServiceDefinition


@dataclass(frozen=True, slots=True)
class OpenAIEndpointProvider:
    base_url: str
    model: str

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")

    def service_definition(self) -> None:
        return None

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "openai_endpoint",
            "base_url": self.base_url,
            "model": self.model,
        }
        return payload


@dataclass(frozen=True, slots=True)
class VLLMProvider:
    model: str
    model_max_context: int | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")
        for key, value in dict(self.extra_kwargs).items():
            if not str(key).strip():
                raise ValueError("extra_kwargs keys must be non-empty")
            if value is None:
                raise ValueError("extra_kwargs values must be non-null")

    def service_definition(self) -> VLLMServiceDefinition:
        return VLLMServiceDefinition(
            model_name_or_path=self.model,
            model_max_context=self.model_max_context,
            extra_kwargs=self.extra_kwargs,
        )

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "type": "vllm",
            "model_name_or_path": self.model,
        }
        if self.model_max_context is not None:
            payload["model_max_context"] = self.model_max_context
        if self.extra_kwargs:
            payload["extra_kwargs"] = dict(self.extra_kwargs)
        return payload


__all__ = ["OpenAIEndpointProvider", "VLLMProvider"]
