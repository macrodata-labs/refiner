from dataclasses import dataclass
from typing import Literal

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
    config: Literal["correctness", "throughput"] = "correctness"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.config not in {"correctness", "throughput"}:
            raise ValueError("config must be 'correctness' or 'throughput'")

    def service_definition(self) -> VLLMServiceDefinition:
        return VLLMServiceDefinition(
            model_name_or_path=self.model,
            config=self.config,
        )

    def to_builtin_args(self) -> dict[str, object]:
        service = self.service_definition()
        payload: dict[str, object] = {
            "type": "vllm",
            **service.to_spec().config,
        }
        return payload


__all__ = ["OpenAIEndpointProvider", "VLLMProvider"]
