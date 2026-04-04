from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

import httpx


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    text: str
    finish_reason: str | None
    usage: Mapping[str, Any]
    response: Mapping[str, Any]


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
    endpoint: str
    headers: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeServiceBinding:
        name = str(payload.get("name", "")).strip()
        kind = str(payload.get("kind", "")).strip()
        endpoint = str(payload.get("endpoint", "")).strip()
        if not name:
            raise ValueError("service binding name must be non-empty")
        if not kind:
            raise ValueError("service binding kind must be non-empty")
        if not endpoint:
            raise ValueError(
                f"service binding {name!r} must include a non-empty endpoint"
            )
        headers = payload.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ValueError(f"service binding {name!r} headers must be an object")
            normalized_headers = {
                str(key): str(value) for key, value in headers.items()
            }
        else:
            normalized_headers = None
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"service binding {name!r} metadata must be an object")
        return cls(
            name=name,
            kind=kind,
            endpoint=endpoint,
            headers=normalized_headers,
            metadata=metadata,
        )


def parse_runtime_service_bindings(
    payload: Mapping[str, Any] | None,
) -> tuple[RuntimeServiceBinding, ...]:
    if payload is None:
        return ()
    services = payload.get("services")
    if services is None:
        return ()
    if not isinstance(services, Sequence):
        raise ValueError("service bindings payload must contain a services list")
    parsed: list[RuntimeServiceBinding] = []
    seen_names: set[str] = set()
    for item in services:
        if not isinstance(item, Mapping):
            raise ValueError("service bindings entries must be objects")
        binding = RuntimeServiceBinding.from_dict(item)
        if binding.name in seen_names:
            raise ValueError(f"duplicate service binding name {binding.name!r}")
        seen_names.add(binding.name)
        parsed.append(binding)
    return tuple(parsed)


class BaseGenerationService(abc.ABC):
    @abc.abstractmethod
    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        raise NotImplementedError


class RuntimeServiceDefinition(abc.ABC):
    name: str
    kind: str

    @abc.abstractmethod
    def to_spec(self) -> RuntimeServiceSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def build_client(
        self, binding: RuntimeServiceBinding | None
    ) -> BaseGenerationService:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _OpenAICompatibleGenerationService(BaseGenerationService):
    endpoint: str
    default_model: str | None = None
    headers: Mapping[str, str] | None = None

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        request_payload = dict(payload)
        if self.default_model is not None and "model" not in request_payload:
            request_payload["model"] = self.default_model
        endpoint_path = (
            "/v1/chat/completions"
            if "messages" in request_payload
            else "/v1/completions"
        )
        async with httpx.AsyncClient(
            base_url=self.endpoint.rstrip("/"),
            timeout=60.0,
            headers=dict(self.headers or {}),
        ) as client:
            response = await client.post(endpoint_path, json=request_payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = ""
            try:
                detail = str(err.response.json())
            except ValueError:
                detail = err.response.text.strip()
            message = f"generation request failed with HTTP {err.response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from err
        response_json = response.json()
        if not isinstance(response_json, Mapping):
            raise RuntimeError("generation response must be a JSON object")
        return _parse_inference_response(
            response_json,
            use_chat="messages" in request_payload,
        )


@dataclass(frozen=True, slots=True)
class LLMServiceDefinition(RuntimeServiceDefinition):
    name: str
    model_name_or_path: str
    model_max_context: int | None = None
    kind: str = "llm"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("service name must be non-empty")
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.model_max_context is not None and self.model_max_context <= 0:
            raise ValueError("model_max_context must be > 0 when provided")

    def to_spec(self) -> RuntimeServiceSpec:
        config: dict[str, Any] = {
            "model_name_or_path": self.model_name_or_path,
        }
        if self.model_max_context is not None:
            config["model_max_context"] = self.model_max_context
        return RuntimeServiceSpec(name=self.name, kind=self.kind, config=config)

    def build_client(
        self, binding: RuntimeServiceBinding | None
    ) -> BaseGenerationService:
        if binding is None:
            raise ValueError(
                f"Service {self.name!r} requires executor-provided runtime bindings. "
                "This service is only supported when the executor provisions runtime services."
            )
        if binding.name != self.name:
            raise ValueError(
                f"service binding name mismatch for {self.name!r}: got {binding.name!r}"
            )
        if binding.kind != self.kind:
            raise ValueError(
                f"service binding kind mismatch for {self.name!r}: "
                f"expected {self.kind!r}, got {binding.kind!r}"
            )
        return _OpenAICompatibleGenerationService(
            endpoint=binding.endpoint,
            default_model=self.model_name_or_path,
            headers=binding.headers,
        )


@dataclass(frozen=True, slots=True)
class LLMEndpointServiceDefinition(RuntimeServiceDefinition):
    name: str
    base_url: str
    api_key_env: str | None = None
    kind: str = "llm_endpoint"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("service name must be non-empty")
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must be non-empty when provided")

    def to_spec(self) -> RuntimeServiceSpec:
        config: dict[str, Any] = {"base_url": self.base_url}
        if self.api_key_env is not None:
            config["api_key_env"] = self.api_key_env
        return RuntimeServiceSpec(name=self.name, kind=self.kind, config=config)

    def build_client(
        self, binding: RuntimeServiceBinding | None
    ) -> BaseGenerationService:
        del binding
        headers: dict[str, str] = {}
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if api_key is None:
                raise ValueError(
                    f"environment variable {self.api_key_env!r} is required "
                    f"for service {self.name!r}"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return _OpenAICompatibleGenerationService(
            endpoint=self.base_url,
            headers=headers or None,
        )


def llm(
    *, name: str, model_name_or_path: str, model_max_context: int | None = None
) -> LLMServiceDefinition:
    return LLMServiceDefinition(
        name=name,
        model_name_or_path=model_name_or_path,
        model_max_context=model_max_context,
    )


def llm_endpoint(
    *, name: str, base_url: str, api_key_env: str | None = None
) -> LLMEndpointServiceDefinition:
    return LLMEndpointServiceDefinition(
        name=name,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _parse_inference_response(
    response_json: Mapping[str, Any], *, use_chat: bool
) -> InferenceResponse:
    choices = response_json.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("generation response is missing choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("generation response choices[0] must be an object")
    if use_chat:
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("chat completion response is missing message")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, Sequence):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if not parts:
                raise RuntimeError(
                    "chat completion response is missing textual content"
                )
            text = "".join(parts)
        else:
            raise RuntimeError("chat completion response is missing textual content")
    else:
        text = choice.get("text")
        if not isinstance(text, str):
            raise RuntimeError("completion response is missing text")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        response=response_json,
    )


__all__ = [
    "BaseGenerationService",
    "InferenceResponse",
    "LLMEndpointServiceDefinition",
    "LLMServiceDefinition",
    "RuntimeServiceBinding",
    "RuntimeServiceDefinition",
    "RuntimeServiceSpec",
    "llm",
    "llm_endpoint",
    "parse_runtime_service_bindings",
]
