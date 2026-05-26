from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from refiner.services import VLLMServiceDefinition


@dataclass(frozen=True, slots=True)
class OpenAIEndpointProvider:
    base_url: str
    model: str
    api_key: str | None = None

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
class GoogleEndpointProvider:
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")

    def service_definition(self) -> None:
        return None

    def to_builtin_args(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "google_endpoint",
            "base_url": self.base_url,
            "model": self.model,
        }
        return payload


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProvider:
    model: str
    base_url: str = "https://api.openai.com"
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")

    def service_definition(self) -> None:
        return None

    def to_builtin_args(self) -> dict[str, object]:
        return {
            "type": "openai_responses",
            "base_url": self.base_url,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class AnthropicEndpointProvider:
    model: str
    base_url: str = "https://api.anthropic.com"
    api_key: str | None = None
    anthropic_version: str = "2023-06-01"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not self.anthropic_version.strip():
            raise ValueError("anthropic_version must be non-empty")

    def service_definition(self) -> None:
        return None

    def to_builtin_args(self) -> dict[str, object]:
        return {
            "type": "anthropic_endpoint",
            "base_url": self.base_url,
            "model": self.model,
            "anthropic_version": self.anthropic_version,
        }


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


__all__ = [
    "AnthropicEndpointProvider",
    "GoogleEndpointProvider",
    "OpenAIEndpointProvider",
    "OpenAIResponsesProvider",
    "VLLMProvider",
]
