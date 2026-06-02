from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlparse

from refiner.inference.internal.media import (
    base64_data,
    is_url,
    parse_data_url,
    resolve_media_type,
)
from refiner.inference.internal.message_conversion import (
    _custom_provider_data,
)
from refiner.inference.internal.schema import StructuredOutputSchema
from refiner.inference.internal.response import (
    InferenceResponse,
    _provider_metadata,
    _text_from_content,
)
from refiner.inference.internal.transport import (
    AiohttpAPIClient,
    post_json_to_api,
    provider_request_options,
)
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
    "thoughtSignature",
}

_ENDPOINT_TIMEOUT_SECONDS = 600.0
_VERTEX_HOST_RE = re.compile(r"^(?:[a-z][a-z0-9-]*-)?aiplatform\.googleapis\.com$")


@dataclass(slots=True)
class _GoogleEndpointClient:
    base_url: str
    model: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None
    _client: AiohttpAPIClient | None = field(default=None, init=False, repr=False)
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

    def _ensure_client(self) -> AiohttpAPIClient:
        client = self._client
        if client is None:
            client = AiohttpAPIClient(
                base_url=self.base_url.rstrip("/"),
                headers=self._resolved_headers,
                timeout_s=_ENDPOINT_TIMEOUT_SECONDS,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        request_payload, max_retries, extra_headers = provider_request_options(payload)
        api_response = await post_json_to_api(
            self._ensure_client(),
            f"{_model_path(self.model)}:generateContent",
            request_payload,
            operation="google generation",
            max_retries=max_retries,
            extra_headers=extra_headers,
        )
        response_json = api_response.value
        if not isinstance(response_json, Mapping):
            raise RuntimeError("google generation response must be a JSON object")
        return parse_response(
            response_json,
            response_headers=api_response.response_headers,
            provider_metadata_name=(
                "googleVertex" if is_vertex_base_url(self.base_url) else "google"
            ),
        )


def _model_path(model: str) -> str:
    return model if "/" in model else f"models/{model}"


def model_capabilities(model: str) -> ModelCapabilities:
    is_gemini = "gemini" in model
    reasoning = model.startswith("gemini-3") or "2.5" in model
    return ModelCapabilities(
        model_family="google",
        images=is_gemini,
        audio=is_gemini,
        video=is_gemini,
        files=is_gemini,
        tools=True,
        structured_output=is_gemini,
        reasoning=reasoning,
        generated_media="image" in model or "flash-image" in model,
        citations=is_gemini,
        known_model=is_gemini,
    )


def build_payload(
    *,
    messages: Sequence[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
    base_url: str = "",
) -> dict[str, Any]:
    is_vertex_provider = is_vertex_base_url(base_url)
    payload = convert_to_google_payload(
        messages,
        generation_config=_generation_config(params),
        provider_options=provider_options,
        is_vertex_provider=is_vertex_provider,
    )
    if schema is not None:
        google_options = _google_options(provider_options, is_vertex_provider)
        structured_outputs = (
            google_options.get("structuredOutputs") if google_options else None
        )
        generation_config = dict(payload.get("generationConfig", {}))
        generation_config["responseMimeType"] = "application/json"
        if structured_outputs is not False:
            converted_schema = _convert_json_schema_to_openapi_schema(
                _inline_json_schema_refs(schema.json_schema)
            )
            if converted_schema is not None:
                generation_config["responseSchema"] = converted_schema
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
    is_vertex_provider: bool = False,
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
                    "parts": _convert_google_assistant_content(
                        content,
                        is_vertex_provider=is_vertex_provider,
                    ),
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
    google_options = _google_options(provider_options, is_vertex_provider)
    if google_options:
        _apply_google_options(
            payload,
            google_options,
            is_vertex_provider=is_vertex_provider,
        )
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


def _convert_google_assistant_content(
    content: object,
    *,
    is_vertex_provider: bool,
) -> list[dict[str, Any]]:
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
                parts.append(
                    _with_google_thought_signature(
                        {"text": part_text},
                        part,
                        is_vertex_provider=is_vertex_provider,
                    )
                )
        elif part_type == "reasoning" and isinstance(part_text, str):
            if part_text:
                parts.append(
                    _with_google_thought_signature(
                        {"text": part_text, "thought": True},
                        part,
                        is_vertex_provider=is_vertex_provider,
                    )
                )
        elif part_type == "file":
            data = part.get("data")
            if is_url(data):
                raise ValueError("google assistant file URLs are not supported")
            media_type = resolve_media_type(
                data,
                declared_media_type=part.get("mediaType"),
            )
            parts.append(
                _with_google_thought_signature(
                    {
                        "inlineData": {
                            "mimeType": media_type,
                            "data": base64_data(data),
                        }
                    },
                    part,
                    is_vertex_provider=is_vertex_provider,
                )
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


def _with_google_thought_signature(
    payload: dict[str, Any],
    part: Mapping[str, Any],
    *,
    is_vertex_provider: bool,
) -> dict[str, Any]:
    google_options = _google_options(part.get("providerOptions"), is_vertex_provider)
    thought_signature = google_options.get("thoughtSignature")
    if isinstance(thought_signature, str):
        payload["thoughtSignature"] = thought_signature
    return payload


def is_vertex_base_url(base_url: str) -> bool:
    hostname = urlparse(base_url).hostname
    if hostname is None:
        hostname = urlparse(f"https://{base_url}").hostname
    if hostname is None:
        return False
    return _VERTEX_HOST_RE.fullmatch(hostname.lower()) is not None


def _google_options(
    provider_options: object,
    is_vertex_provider: bool,
) -> Mapping[str, Any]:
    if not isinstance(provider_options, Mapping):
        return {}
    options_by_namespace = cast(Mapping[str, Any], provider_options)
    names = ("googleVertex", "vertex", "google") if is_vertex_provider else ("google",)
    for name in names:
        options = options_by_namespace.get(name)
        if isinstance(options, Mapping):
            return options
    return {}


def _apply_google_options(
    payload: dict[str, Any],
    options: Mapping[str, Any],
    *,
    is_vertex_provider: bool,
) -> None:
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
        "systemInstruction",
    ):
        if key in options:
            payload[key] = options[key]
    if is_vertex_provider:
        headers = dict(payload.get("__refiner_headers", {}))
        if "sharedRequestType" in options:
            headers["X-Vertex-AI-LLM-Shared-Request-Type"] = str(
                options["sharedRequestType"]
            )
        if "requestType" in options:
            headers["X-Vertex-AI-LLM-Request-Type"] = str(options["requestType"])
        if headers:
            payload["__refiner_headers"] = headers
    elif "serviceTier" in options:
        payload["serviceTier"] = options["serviceTier"]


def _inline_json_schema_refs(json_schema: Mapping[str, Any]) -> dict[str, Any]:
    definitions = {
        **_mapping_at(json_schema, "$defs"),
        **_mapping_at(json_schema, "definitions"),
    }

    def _visit(value: Any, resolving: frozenset[str]) -> Any:
        if isinstance(value, Mapping):
            ref = value.get("$ref")
            if isinstance(ref, str):
                if ref in resolving:
                    raise ValueError(
                        "cyclic structured output schema refs are not supported"
                    )
                target = _resolve_local_ref(ref, definitions)
                if target is not None:
                    merged = {key: item for key, item in value.items() if key != "$ref"}
                    resolved = _visit(target, resolving | {ref})
                    if isinstance(resolved, dict):
                        return {**resolved, **_visit(merged, resolving)}
                    return resolved
            return {key: _visit(item, resolving) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return [_visit(item, resolving) for item in value]
        return value

    inlined = _visit(json_schema, frozenset())
    if not isinstance(inlined, dict):
        raise ValueError("structured output schema must be a JSON object")
    inlined.pop("$defs", None)
    inlined.pop("definitions", None)
    return inlined


def _mapping_at(source: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_local_ref(
    ref: str,
    definitions: Mapping[str, Any],
) -> Any | None:
    if ref.startswith("#/$defs/"):
        return definitions.get(ref.removeprefix("#/$defs/"))
    if ref.startswith("#/definitions/"):
        return definitions.get(ref.removeprefix("#/definitions/"))
    return None


def _convert_json_schema_to_openapi_schema(
    json_schema: Any,
    *,
    is_root: bool = True,
) -> dict[str, Any] | None:
    if json_schema is None:
        return None
    if _is_empty_object_schema(json_schema):
        if is_root:
            return None
        result = {"type": "object"}
        description = json_schema.get("description")
        if isinstance(description, str):
            result["description"] = description
        return result
    if isinstance(json_schema, bool):
        return {"type": "boolean", "properties": {}}
    if not isinstance(json_schema, Mapping):
        return None

    result: dict[str, Any] = {}
    for key in (
        "description",
        "required",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "propertyOrdering",
        "additionalProperties",
    ):
        if key in json_schema:
            result[key] = json_schema[key]

    const_value = json_schema.get("const")
    if const_value is not None:
        result["enum"] = [const_value]

    schema_type = json_schema.get("type")
    if isinstance(schema_type, Sequence) and not isinstance(schema_type, str):
        has_null = "null" in schema_type
        non_null_types = [item for item in schema_type if item != "null"]
        if not non_null_types:
            result["type"] = "null"
        else:
            result["anyOf"] = [{"type": item} for item in non_null_types]
            if has_null:
                result["nullable"] = True
    elif schema_type is not None:
        result["type"] = schema_type

    if "enum" in json_schema:
        result["enum"] = json_schema["enum"]

    properties = json_schema.get("properties")
    if isinstance(properties, Mapping):
        result["properties"] = {
            key: _convert_json_schema_to_openapi_schema(value, is_root=False)
            for key, value in properties.items()
        }

    items = json_schema.get("items")
    if isinstance(items, Sequence) and not isinstance(items, Mapping | str | bytes):
        result["items"] = [
            _convert_json_schema_to_openapi_schema(item, is_root=False)
            for item in items
        ]
    elif items is not None:
        result["items"] = _convert_json_schema_to_openapi_schema(
            items,
            is_root=False,
        )

    prefix_items = json_schema.get("prefixItems")
    if isinstance(prefix_items, Sequence) and not isinstance(prefix_items, str | bytes):
        result["prefixItems"] = [
            _convert_json_schema_to_openapi_schema(item, is_root=False)
            for item in prefix_items
        ]

    for key in ("allOf", "oneOf"):
        value = json_schema.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            result[key] = [
                _convert_json_schema_to_openapi_schema(item, is_root=False)
                for item in value
            ]

    any_of = json_schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, str | bytes):
        null_schemas = [
            schema
            for schema in any_of
            if isinstance(schema, Mapping) and schema.get("type") == "null"
        ]
        non_null_schemas = [schema for schema in any_of if schema not in null_schemas]
        if null_schemas and len(non_null_schemas) == 1:
            converted = _convert_json_schema_to_openapi_schema(
                non_null_schemas[0],
                is_root=False,
            )
            if converted is not None:
                result.update(converted)
                result["nullable"] = True
        else:
            result["anyOf"] = [
                _convert_json_schema_to_openapi_schema(item, is_root=False)
                for item in non_null_schemas
            ]
            if null_schemas:
                result["nullable"] = True

    return result


def _is_empty_object_schema(json_schema: Any) -> bool:
    return (
        isinstance(json_schema, Mapping)
        and json_schema.get("type") == "object"
        and not json_schema.get("properties")
        and not json_schema.get("additionalProperties")
    )


def parse_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
    provider_metadata_name: str = "google",
) -> InferenceResponse:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        prompt_feedback = response_json.get("promptFeedback")
        if isinstance(prompt_feedback, Mapping):
            block_reason = prompt_feedback.get("blockReason")
            if block_reason is not None:
                raise RuntimeError(
                    "google generation response is missing candidates[0]: "
                    f"promptFeedback.blockReason={block_reason}"
                )
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
            metadata = _google_part_metadata(part, provider_metadata_name)
            if part.get("thought") is True:
                content_parts.append(
                    {"type": "reasoning", "text": part["text"], **metadata}
                )
            else:
                content_parts.append({"type": "text", "text": part["text"], **metadata})
        content_parts.extend(_google_generated_parts(part, provider_metadata_name))
    content_parts.extend(_google_grounding_sources(candidate, provider_metadata_name))
    text = _text_from_content(content_parts)
    if not text and not content_parts:
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
        provider_metadata=_google_provider_metadata(
            response_json,
            candidate,
            provider_metadata_name,
        ),
    )


def _google_part_metadata(part: Mapping[str, Any], provider: str) -> dict[str, Any]:
    metadata = {}
    for key in ("thoughtSignature", "thought", "executableCode", "codeExecutionResult"):
        if key in part:
            metadata[key] = part[key]
    return {"providerMetadata": {provider: metadata}} if metadata else {}


def _google_provider_metadata(
    response_json: Mapping[str, Any],
    candidate: Mapping[str, Any],
    provider: str,
) -> dict[str, Mapping[str, Any]]:
    metadata = dict(
        _provider_metadata(provider, response_json, candidate).get(provider, {})
    )
    for key in (
        "promptFeedback",
        "usageMetadata",
        "modelVersion",
        "responseId",
    ):
        if key in response_json:
            metadata[key] = response_json[key]
    for key in (
        "groundingMetadata",
        "urlContextMetadata",
        "safetyRatings",
        "finishMessage",
        "citationMetadata",
    ):
        if key in candidate:
            metadata[key] = candidate[key]
    usage = response_json.get("usageMetadata")
    if isinstance(usage, Mapping) and "serviceTier" in usage:
        metadata["serviceTier"] = usage["serviceTier"]
    return {provider: metadata} if metadata else {}


def _google_generated_parts(
    part: Mapping[str, Any],
    provider: str,
) -> list[ResponseContentPart]:
    inline_data = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline_data, Mapping):
        media_type = inline_data.get("mimeType") or inline_data.get("mime_type")
        data = inline_data.get("data")
        top_level = media_type.split("/", 1)[0] if isinstance(media_type, str) else ""
        result: dict[str, Any] = {
            "type": "image" if top_level == "image" else "file",
            "providerMetadata": {provider: dict(part)},
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
            "providerMetadata": {provider: dict(part)},
        }
        if isinstance(media_type, str):
            result["mediaType"] = media_type
        if isinstance(url, str):
            result["url"] = url
        return [cast(ResponseContentPart, result)]
    return []


def _google_grounding_sources(
    candidate: Mapping[str, Any],
    provider: str,
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
            "providerMetadata": {provider: dict(chunk)},
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
    "is_vertex_base_url",
    "model_capabilities",
    "parse_response",
]
