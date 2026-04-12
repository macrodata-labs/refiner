from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")

    def service_definition(self) -> VLLMServiceDefinition:
        return VLLMServiceDefinition(
            model_name_or_path=self.model,
            model_max_context=self.model_max_context,
        )

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "type": "vllm",
            "model_name_or_path": self.model,
        }
        if self.model_max_context is not None:
            payload["model_max_context"] = self.model_max_context
        return payload


__all__ = ["OpenAIEndpointProvider", "VLLMProvider"]
