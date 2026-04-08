from dataclasses import dataclass
from typing import Any

from refiner.services import VLLMServiceDefinition


@dataclass(frozen=True, slots=True)
class OpenAIEndpointProvider:
    base_url: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")

    def service_definition(self) -> None:
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
    modal_mode: str = "endpoint"

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")
        if self.modal_mode not in {"endpoint", "function"}:
            raise ValueError("modal_mode must be one of {'endpoint', 'function'}")

    def service_definition(self) -> VLLMServiceDefinition:
        return VLLMServiceDefinition(
            model_name_or_path=self.model_name_or_path,
            model_max_context=self.model_max_context,
            modal_mode=self.modal_mode,
        )

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, Any] = {
            "type": "vllm",
            "model_name_or_path": self.model_name_or_path,
        }
        if self.model_max_context is not None:
            payload["model_max_context"] = self.model_max_context
        if self.modal_mode != "endpoint":
            payload["modal_mode"] = self.modal_mode
        return payload


__all__ = ["OpenAIEndpointProvider", "VLLMProvider"]
