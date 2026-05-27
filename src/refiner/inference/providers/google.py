from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from refiner.inference._media import (
    base64_data,
    is_url,
    parse_data_url,
    resolve_media_type,
)
from refiner.inference._message_conversion import (
    _custom_provider_data,
    _provider_options,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference._response import (
    InferenceResponse,
    _provider_metadata,
    _text_from_content,
)
from refiner.inference._transport import post_json_to_api
from refiner.inference.types import (
    Message,
    ModelCapabilities,
    ProviderOptions,
    ResponseContentPart,
)

PROVIDER_OPTIONS = {
    "apiClient",
    "responseModalities",
    "responseMimeType",
    "responseSchema",
    "thinkingConfig",
    "cachedContent",
    "structuredOutputs",
    "safetySettings",
    "threshold",
    "audioTimestamp",
    "labels",
    "mediaResolution",
    "imageConfig",
    "retrievalConfig",
    "speechConfig",
    "systemInstruction",
    "streamFunctionCallArguments",
    "serviceTier",
    "sharedRequestType",
    "requestType",
}

_ENDPOINT_TIMEOUT_SECONDS = 600.0


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
                timeout=_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        api_response = await post_json_to_api(
            self._ensure_client(),
            f"{_model_path(self.model)}:generateContent",
            _request_payload(payload),
            operation="google generation",
            max_retries=_max_retries(payload),
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("google generation response must be a JSON object")
        return parse_response(
            response_json,
            response_headers=api_response.response_headers,
        )


def _model_path(model: str) -> str:
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


def model_capabilities(model: str) -> ModelCapabilities:
    is_gemini = "gemini" in model
    return ModelCapabilities(
        images=is_gemini,
        audio=is_gemini,
        video=is_gemini,
        files=is_gemini,
        tools=True,
        structured_output=is_gemini,
        reasoning="2.5" in model or "3" in model,
        generated_media="image" in model or "flash-image" in model,
        citations=is_gemini,
    )


def build_payload(
    *,
    messages: list[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = convert_to_google_payload(
        messages,
        generation_config=_generation_config(params),
        provider_options=provider_options,
    )
    if schema is not None:
        generation_config = dict(payload.get("generationConfig", {}))
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema.json_schema
        payload["generationConfig"] = generation_config
    return payload


def _generation_config(params: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(params)
    aliases = {
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "stop_sequences": "stopSequences",
    }
    for source, target in aliases.items():
        if source in config and target not in config:
            config[target] = config.pop(source)
    return config


def convert_to_google_payload(
    messages: Sequence[Message],
    *,
    generation_config: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    system_messages_allowed = True

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            if not isinstance(content, str):
                raise ValueError("system message content must be a string")
            if not system_messages_allowed:
                raise ValueError(
                    "google system messages are only supported before user messages"
                )
            system_parts.append({"text": content})
            continue
        system_messages_allowed = False
        if role == "assistant":
            contents.append(
                {
                    "role": "model",
                    "parts": _convert_google_assistant_content(content),
                }
            )
            continue
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = [_convert_google_user_part(part) for part in content]
        contents.append({"role": "user", "parts": parts})

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if generation_config:
        payload["generationConfig"] = dict(generation_config)
    google_options = _provider_options(provider_options, "google")
    if google_options:
        _apply_google_options(payload, google_options)
    return payload


def _convert_google_user_part(part: Mapping[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"google"})
    if part_type == "text":
        return {"text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _google_file_part(
            image_data,
            resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
        )
    if part_type == "file":
        file_data = part.get("data")
        return _google_file_part(
            file_data,
            resolve_media_type(file_data, declared_media_type=part.get("mediaType")),
        )
    raise ValueError(f"unsupported content part type {part_type!r}")


def _convert_google_assistant_content(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            if part_text:
                parts.append({"text": part_text})
        elif part_type == "reasoning" and isinstance(part_text, str):
            if part_text:
                parts.append({"text": part_text, "thought": True})
        elif part_type == "file":
            data = part.get("data")
            if is_url(data):
                raise ValueError("google assistant file URLs are not supported")
            media_type = resolve_media_type(
                data,
                declared_media_type=part.get("mediaType"),
            )
            parts.append(
                {
                    "inlineData": {
                        "mimeType": media_type,
                        "data": base64_data(data),
                    }
                }
            )
        elif part_type == "custom":
            parts.append(_custom_provider_data(part, {"google"}))
    return parts


def _google_file_part(data: object, media_type: str) -> dict[str, Any]:
    parsed = parse_data_url(data)
    if parsed is not None:
        parsed_media_type, parsed_data = parsed
        return {"inlineData": {"mimeType": parsed_media_type, "data": parsed_data}}
    if is_url(data):
        return {"fileData": {"mimeType": media_type, "fileUri": str(data)}}
    return {"inlineData": {"mimeType": media_type, "data": base64_data(data)}}


def _apply_google_options(payload: dict[str, Any], options: Mapping[str, Any]) -> None:
    generation_keys = {
        "audioTimestamp",
        "imageConfig",
        "mediaResolution",
        "responseMimeType",
        "responseModalities",
        "responseSchema",
        "speechConfig",
        "thinkingConfig",
    }
    generation_config = payload.setdefault("generationConfig", {})
    if isinstance(generation_config, dict):
        for key in generation_keys:
            if key in options:
                generation_config[key] = options[key]
    for key in (
        "cachedContent",
        "labels",
        "retrievalConfig",
        "safetySettings",
        "serviceTier",
        "systemInstruction",
    ):
        if key in options:
            payload[key] = options[key]


def parse_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise RuntimeError("google generation response is missing candidates[0]")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise RuntimeError("google generation response candidates[0] must be an object")
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        raise RuntimeError("google generation response is missing content")
    parts = content.get("parts")
    if not isinstance(parts, Sequence):
        raise RuntimeError("google generation response is missing content.parts")
    content_parts: list[ResponseContentPart] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if isinstance(part.get("text"), str):
            if part.get("thought") is True:
                content_parts.append({"type": "reasoning", "text": part["text"]})
            else:
                content_parts.append({"type": "text", "text": part["text"]})
        content_parts.extend(_google_generated_parts(part))
    content_parts.extend(_google_grounding_sources(candidate))
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("google generation response is missing textual content")
    usage_metadata = response_json.get("usageMetadata")
    usage = _google_usage(usage_metadata if isinstance(usage_metadata, Mapping) else {})
    finish_reason = candidate.get("finishReason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
        provider_metadata=_provider_metadata("google", response_json, candidate),
    )


def _google_generated_parts(part: Mapping[str, Any]) -> list[ResponseContentPart]:
    inline_data = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline_data, Mapping):
        media_type = inline_data.get("mimeType") or inline_data.get("mime_type")
        data = inline_data.get("data")
        top_level = media_type.split("/", 1)[0] if isinstance(media_type, str) else ""
        result: dict[str, Any] = {
            "type": "image" if top_level == "image" else "file",
            "providerMetadata": {"google": dict(part)},
        }
        if isinstance(media_type, str):
            result["mediaType"] = media_type
        if isinstance(data, str):
            result["data"] = data
        return [cast(ResponseContentPart, result)]
    file_data = part.get("fileData") or part.get("file_data")
    if isinstance(file_data, Mapping):
        media_type = file_data.get("mimeType") or file_data.get("mime_type")
        url = file_data.get("fileUri") or file_data.get("file_uri")
        top_level = media_type.split("/", 1)[0] if isinstance(media_type, str) else ""
        result = {
            "type": "image" if top_level == "image" else "file",
            "providerMetadata": {"google": dict(part)},
        }
        if isinstance(media_type, str):
            result["mediaType"] = media_type
        if isinstance(url, str):
            result["url"] = url
        return [cast(ResponseContentPart, result)]
    return []


def _google_grounding_sources(
    candidate: Mapping[str, Any],
) -> list[ResponseContentPart]:
    grounding = candidate.get("groundingMetadata")
    if not isinstance(grounding, Mapping):
        return []
    chunks = grounding.get("groundingChunks")
    if not isinstance(chunks, Sequence) or isinstance(chunks, str):
        return []
    sources: list[ResponseContentPart] = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        web = chunk.get("web")
        if not isinstance(web, Mapping):
            continue
        url = web.get("uri")
        if not isinstance(url, str):
            continue
        source: dict[str, Any] = {
            "type": "source",
            "sourceType": "url",
            "url": url,
            "providerMetadata": {"google": dict(chunk)},
        }
        title = web.get("title")
        if isinstance(title, str):
            source["title"] = title
        sources.append(cast(ResponseContentPart, source))
    return sources


def _google_usage(usage_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    usage: dict[str, Any] = {}
    if "promptTokenCount" in usage_metadata:
        usage["prompt_tokens"] = usage_metadata["promptTokenCount"]
    if "candidatesTokenCount" in usage_metadata:
        usage["completion_tokens"] = usage_metadata["candidatesTokenCount"]
    if "totalTokenCount" in usage_metadata:
        usage["total_tokens"] = usage_metadata["totalTokenCount"]
    return usage


__all__ = [
    "PROVIDER_OPTIONS",
    "_GoogleEndpointClient",
    "build_payload",
    "model_capabilities",
    "parse_response",
]
