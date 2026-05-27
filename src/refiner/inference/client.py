from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from refiner.inference._response import InferenceResponse
from refiner.inference._transport import post_json_to_api
from refiner.inference.providers import anthropic as anthropic_provider
from refiner.inference.providers import google as google_provider
from refiner.inference.providers import openai as openai_provider

_OPENAI_ENDPOINT_TIMEOUT_SECONDS = 600.0


@dataclass(slots=True)
class _OpenAIEndpointClient:
    base_url: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("OPENAI_API_KEY")
        if resolved_api_key is not None:
            headers["Authorization"] = f"Bearer {resolved_api_key}"
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_normalize_openai_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        use_chat = "messages" in payload
        endpoint_path = "v1/chat/completions" if use_chat else "v1/completions"
        api_response = await post_json_to_api(
            self._ensure_client(),
            endpoint_path,
            _request_payload(payload),
            operation="generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("generation response must be a JSON object")
        return openai_provider.parse_chat_response(
            response_json,
            use_chat=use_chat,
            response_headers=api_response.response_headers,
        )

    async def pooling(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "pooling",
            _request_payload(payload),
            operation="pooling",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("pooling response must be a JSON object")
        return response_json


@dataclass(slots=True)
class _GoogleEndpointClient:
    base_url: str
    model: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        if resolved_api_key is not None:
            headers["x-goog-api-key"] = resolved_api_key
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            f"{_google_model_path(self.model)}:generateContent",
            _request_payload(payload),
            operation="google generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("google generation response must be a JSON object")
        return google_provider.parse_response(
            response_json,
            response_headers=api_response.response_headers,
        )


@dataclass(slots=True)
class _OpenAIResponsesClient:
    base_url: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("OPENAI_API_KEY")
        if resolved_api_key is not None:
            headers["Authorization"] = f"Bearer {resolved_api_key}"
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=_normalize_openai_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "v1/responses",
            _request_payload(payload),
            operation="openai responses generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("openai responses response must be a JSON object")
        return openai_provider.parse_responses_response(
            response_json,
            response_headers=api_response.response_headers,
        )


@dataclass(slots=True)
class _AnthropicEndpointClient:
    base_url: str
    api_key: str | None = None
    anthropic_version: str = "2023-06-01"
    headers: Mapping[str, str] | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _resolved_headers: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        headers = dict(self.headers or {})
        resolved_api_key = self.api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if resolved_api_key is not None:
            headers["x-api-key"] = resolved_api_key
        headers["anthropic-version"] = self.anthropic_version
        self._resolved_headers = headers

    def _ensure_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=self._resolved_headers,
                timeout=_OPENAI_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            "v1/messages",
            _request_payload(payload),
            operation="anthropic generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("anthropic generation response must be a JSON object")
        return anthropic_provider.parse_response(
            response_json,
            response_headers=api_response.response_headers,
        )


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _google_model_path(model: str) -> str:
    return model if "/" in model else f"models/{model}"


def _max_retries(payload: Mapping[str, Any]) -> int | None:
    raw = payload.get("__refiner_max_retries")
    if raw is None:
        return None
    if not isinstance(raw, int):
        raise ValueError("maxRetries must be an integer")
    return raw


def _request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    request.pop("__refiner_max_retries", None)
    return request


__all__ = ["InferenceResponse"]
