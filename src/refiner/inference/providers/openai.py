from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from refiner.inference._media import (
    base64_data,
    data_or_url,
    is_url,
    resolve_media_type,
    top_level_media_type,
)
from refiner.inference._capabilities import ModelCapabilities
from refiner.inference._message_conversion import (
    _custom_provider_data,
    _provider_option,
)
from refiner.inference._schema import StructuredOutputSchema
from refiner.inference._response import (
    InferenceResponse,
    _copy_if_str,
    _provider_metadata,
    _text_from_content,
)
from refiner.inference._transport import post_json_to_api
from refiner.inference.types import Message, ProviderOptions, ResponseContentPart

logger = logging.getLogger(__name__)

CHAT_PROVIDER_OPTIONS = {
    "audio",
    "background",
    "logitBias",
    "logprobs",
    "modalities",
    "parallelToolCalls",
    "user",
    "responseFormat",
    "reasoningEffort",
    "maxCompletionTokens",
    "store",
    "metadata",
    "prediction",
    "serviceTier",
    "reasoningSummary",
    "textVerbosity",
    "promptCacheKey",
    "promptCacheRetention",
    "safetyIdentifier",
    "text",
    "topLogprobs",
    "webSearchOptions",
}

RESPONSES_PROVIDER_OPTIONS = {
    *CHAT_PROVIDER_OPTIONS,
    "conversation",
    "include",
    "instructions",
    "maxToolCalls",
    "previousResponseId",
    "truncation",
    "contextManagement",
}

_ENDPOINT_TIMEOUT_SECONDS = 600.0


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
                base_url=_normalize_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_ENDPOINT_TIMEOUT_SECONDS,
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
        return parse_chat_response(
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
                base_url=_normalize_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout=_ENDPOINT_TIMEOUT_SECONDS,
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
        return parse_responses_response(
            response_json,
            response_headers=api_response.response_headers,
        )


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


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


def model_capabilities(model: str, *, responses_api: bool) -> ModelCapabilities:
    vision = any(
        marker in model for marker in ("gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4", "omni")
    )
    reasoning = model.startswith(("o1", "o3", "o4")) or "gpt-5" in model
    return ModelCapabilities(
        images=vision,
        audio="audio" in model or "realtime" in model,
        video=False,
        files=responses_api,
        tools=True,
        structured_output=True,
        reasoning=reasoning,
        generated_media="image" in model,
        citations=responses_api,
    )


def build_chat_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["messages"] = convert_to_openai_chat_messages(messages)
    elif schema is None:
        payload["prompt"] = prompt
    else:
        payload["messages"] = [{"role": "user", "content": prompt or ""}]
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": _json_schema(schema),
        }
    if provider_options is not None:
        payload["providerOptions"] = provider_options
    return payload


def build_responses_payload(
    *,
    messages: Sequence[Message] | None,
    prompt: str | None,
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    if messages is not None:
        payload["input"] = convert_to_openai_responses_input(messages)
    else:
        payload["input"] = prompt
    _apply_responses_options(payload, provider_options)
    _normalize_responses_params(payload)
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        text = dict(payload.get("text", {}))
        text["format"] = _response_format(schema)
        payload["text"] = text
    return payload


def _apply_responses_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    passthrough = {
        "conversation",
        "background",
        "include",
        "instructions",
        "logprobs",
        "maxToolCalls",
        "metadata",
        "parallelToolCalls",
        "previousResponseId",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "serviceTier",
        "store",
        "truncation",
        "user",
        "contextManagement",
        "text",
        "topLogprobs",
        "webSearchOptions",
    }
    for key in passthrough:
        if key in openai_options:
            payload[key] = openai_options[key]


def _normalize_responses_params(payload: dict[str, Any]) -> None:
    aliases = {
        "max_tokens": "max_output_tokens",
        "maxCompletionTokens": "max_output_tokens",
    }
    for source, target in aliases.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)


def _normalize_reasoning_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    for key in (
        "logitBias",
        "logprobs",
        "topLogprobs",
        "parallelToolCalls",
        "user",
        "maxCompletionTokens",
        "modalities",
        "audio",
        "store",
        "metadata",
        "prediction",
        "serviceTier",
        "promptCacheKey",
        "promptCacheRetention",
        "safetyIdentifier",
        "webSearchOptions",
        "responseFormat",
    ):
        if key in openai_options:
            payload[key] = openai_options[key]
    reasoning: dict[str, Any] = {}
    if "reasoningEffort" in openai_options:
        reasoning["effort"] = openai_options["reasoningEffort"]
    if "reasoningSummary" in openai_options:
        reasoning["summary"] = openai_options["reasoningSummary"]
    if reasoning:
        payload["reasoning"] = {**dict(payload.get("reasoning", {})), **reasoning}


def _normalize_text_options(
    payload: dict[str, Any],
    provider_options: ProviderOptions | None,
) -> None:
    if not provider_options:
        return
    openai_options = provider_options.get("openai", {})
    if "textVerbosity" in openai_options:
        text = dict(payload.get("text", {}))
        text["verbosity"] = openai_options["textVerbosity"]
        payload["text"] = text


def _response_format(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


def _json_schema(schema: StructuredOutputSchema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "schema": schema.json_schema,
        "strict": schema.strict,
    }


def convert_to_openai_chat_messages(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append({"role": role, "content": content})
            continue
        if role == "assistant":
            converted.append(
                {
                    "role": "assistant",
                    "content": _convert_openai_chat_assistant_content(content),
                }
            )
            continue
        if isinstance(content, str):
            converted.append({"role": "user", "content": content})
            continue
        if len(content) == 1 and content[0]["type"] == "text":
            converted.append({"role": "user", "content": content[0]["text"]})
            continue
        converted.append(
            {
                "role": "user",
                "content": [
                    _convert_openai_user_part(part, index)
                    for index, part in enumerate(content)
                ],
            }
        )
    return converted


def convert_to_openai_responses_input(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            input_items.append({"role": role, "content": content})
            continue
        if role == "assistant":
            input_items.extend(_convert_openai_responses_assistant_content(content))
            continue
        if isinstance(content, str):
            input_items.append({"role": "user", "content": content})
            continue
        input_items.append(
            {
                "role": "user",
                "content": [
                    _convert_openai_responses_part(part, index)
                    for index, part in enumerate(content)
                ],
            }
        )
    return input_items


def _convert_openai_user_part(part: Mapping[str, Any], index: int) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"openai", "openai-chat"})
    if part_type == "text":
        return {"type": "text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        return _openai_image_url_part(
            image_data,
            media_type=resolve_media_type(
                image_data,
                declared_media_type=part.get("mediaType"),
                default_top_level="image",
            ),
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = resolve_media_type(
        data,
        declared_media_type=part.get("mediaType"),
    )
    top_level = top_level_media_type(media_type)
    if top_level == "image":
        return _openai_image_url_part(
            data,
            media_type=media_type,
            detail=_provider_option(part, "openai", "imageDetail"),
        )
    if top_level == "audio":
        if is_url(data):
            raise ValueError("openai audio file parts with URLs are not supported")
        full_media_type = resolve_media_type(
            data,
            declared_media_type=media_type,
            default_top_level="audio",
        )
        if full_media_type in {"audio/wav", "audio/x-wav"}:
            audio_format = "wav"
        elif full_media_type in {"audio/mp3", "audio/mpeg"}:
            audio_format = "mp3"
        else:
            raise ValueError(
                f"openai audio content parts with media type {full_media_type} "
                "are not supported"
            )
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64_data(data),
                "format": audio_format,
            },
        }
    full_media_type = resolve_media_type(data, declared_media_type=media_type)
    if full_media_type != "application/pdf":
        raise ValueError(
            f"openai file part media type {full_media_type} is unsupported"
        )
    if is_url(data):
        raise ValueError("openai PDF file parts with URLs are not supported")
    return {
        "type": "file",
        "file": {
            "filename": part.get("filename", f"part-{index}.pdf"),
            "file_data": f"data:application/pdf;base64,{base64_data(data)}",
        },
    }


def _convert_openai_responses_part(
    part: Mapping[str, Any], index: int
) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "custom":
        return _custom_provider_data(part, {"openai", "openai-responses"})
    if part_type == "text":
        return {"type": "input_text", "text": part["text"]}
    if part_type == "image":
        image_data = part.get("image")
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": data_or_url(
                image_data,
                resolve_media_type(
                    image_data,
                    declared_media_type=part.get("mediaType"),
                    default_top_level="image",
                ),
            ),
        }
        detail = _provider_option(part, "openai", "imageDetail")
        if detail is not None:
            image_part["detail"] = detail
        return image_part
    if part_type != "file":
        raise ValueError(f"unsupported content part type {part_type!r}")

    data = part.get("data")
    media_type = resolve_media_type(data, declared_media_type=part.get("mediaType"))
    if top_level_media_type(media_type) == "image":
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": data_or_url(data, media_type),
        }
        detail = _provider_option(part, "openai", "imageDetail")
        if detail is not None:
            image_part["detail"] = detail
        return image_part
    if is_url(data):
        return {"type": "input_file", "file_url": str(data)}
    full_media_type = resolve_media_type(data, declared_media_type=media_type)
    return {
        "type": "input_file",
        "filename": part.get(
            "filename",
            f"part-{index}.pdf"
            if full_media_type == "application/pdf"
            else f"part-{index}",
        ),
        "file_data": f"data:{full_media_type};base64,{base64_data(data)}",
    }


def _openai_image_url_part(
    data: object,
    *,
    media_type: object,
    detail: object,
) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": data_or_url(data, str(media_type))}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _convert_openai_chat_assistant_content(
    content: object,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    text_parts: list[str] = []
    converted_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            text_parts.append(part_text)
            converted_parts.append({"type": "text", "text": part_text})
        elif part_type == "custom":
            converted_parts.append(
                _custom_provider_data(part, {"openai", "openai-chat"})
            )
        elif part_type in {"file", "reasoning"}:
            raise ValueError(
                f"openai chat assistant {part_type} parts are not supported"
            )
    if any(part.get("type") != "text" for part in converted_parts):
        return converted_parts
    return "".join(text_parts)


def _convert_openai_responses_assistant_content(
    content: object,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]
    if not isinstance(content, Sequence):
        raise ValueError("assistant message content must be a string or content parts")
    input_items: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part = dict(part)
        part_type = part.get("type")
        part_text = part.get("text")
        if part_type == "text" and isinstance(part_text, str):
            text_parts.append(part_text)
        elif part_type == "reasoning" and isinstance(part_text, str):
            input_items.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": part_text}],
                }
            )
        elif part_type == "file":
            raise ValueError("openai responses assistant file parts are not supported")
        elif part_type == "custom":
            input_items.append(
                _custom_provider_data(part, {"openai", "openai-responses"})
            )
    if text_parts:
        input_items.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(text_parts)}],
            }
        )
    return input_items


def parse_chat_response(
    response_json: Mapping[str, Any],
    *,
    use_chat: bool,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    choices = response_json.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("generation response is missing choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("generation response choices[0] must be an object")
    content_parts: list[ResponseContentPart] = []
    if use_chat:
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("chat completion response is missing message")
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            content_parts.append({"type": "reasoning", "text": reasoning})
        content_parts.extend(_openai_sources(message.get("annotations")))
        content = message.get("content")
        if isinstance(content, str):
            text = content
            if text:
                content_parts.append({"type": "text", "text": text})
        elif content is None:
            logger.warning(
                "chat completion response had null message.content; returning empty text",
                extra={
                    "finish_reason": choice.get("finish_reason"),
                    "has_reasoning": isinstance(message.get("reasoning"), str),
                },
            )
            text = ""
        elif isinstance(content, Sequence):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    content_parts.extend(_openai_sources(item.get("annotations")))
            if not parts:
                raise RuntimeError(
                    "chat completion response is missing textual content"
                )
            text = "".join(parts)
            for part in parts:
                content_parts.append({"type": "text", "text": part})
        else:
            raise RuntimeError("chat completion response is missing textual content")
    else:
        text = choice.get("text")
        if not isinstance(text, str):
            raise RuntimeError("completion response is missing text")
        if text:
            content_parts.append({"type": "text", "text": text})
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
        content=content_parts,
        headers=dict(response_headers or {}),
        logprobs=_sequence_or_empty(choice.get("logprobs")),
        provider_metadata=_provider_metadata("openai", response_json, choice),
    )


def parse_responses_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    content_parts: list[ResponseContentPart] = []
    output = response_json.get("output")
    if isinstance(output, Sequence):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, Sequence):
                    for summary_part in summary:
                        if isinstance(summary_part, Mapping) and isinstance(
                            summary_part.get("text"), str
                        ):
                            content_parts.append(
                                {"type": "reasoning", "text": summary_part["text"]}
                            )
                continue
            content = item.get("content")
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if isinstance(part.get("text"), str):
                    content_parts.append({"type": "text", "text": part["text"]})
                    content_parts.extend(_openai_sources(part.get("annotations")))
                content_parts.extend(_openai_generated_parts(part))
    if not _text_from_content(content_parts) and isinstance(
        response_json.get("output_text"), str
    ):
        content_parts.append({"type": "text", "text": response_json["output_text"]})
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("openai responses response is missing textual content")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    mapped_usage = {
        "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "completion_tokens": usage.get(
            "output_tokens", usage.get("completion_tokens", 0)
        ),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return InferenceResponse(
        text=text,
        finish_reason=None,
        usage=mapped_usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
        logprobs=_collect_openai_responses_logprobs(response_json),
        provider_metadata=_provider_metadata("openai", response_json),
    )


def _openai_sources(annotations: object) -> list[ResponseContentPart]:
    if not isinstance(annotations, Sequence) or isinstance(annotations, str):
        return []
    sources: list[ResponseContentPart] = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        annotation = cast(Mapping[str, Any], annotation)
        url = annotation.get("url")
        title = annotation.get("title")
        if not isinstance(url, str):
            url_citation = annotation.get("url_citation")
            if isinstance(url_citation, Mapping):
                url = url_citation.get("url")
                title = url_citation.get("title", title)
        if isinstance(url, str):
            source: dict[str, Any] = {
                "type": "source",
                "sourceType": "url",
                "url": url,
                "providerMetadata": {"openai": dict(annotation)},
            }
            if isinstance(title, str):
                source["title"] = title
            sources.append(cast(ResponseContentPart, source))
    return sources


def _openai_generated_parts(part: Mapping[str, Any]) -> list[ResponseContentPart]:
    part_type = part.get("type")
    if part_type in {"output_image", "image"}:
        image: dict[str, Any] = {
            "type": "image",
            "providerMetadata": {"openai": dict(part)},
        }
        _copy_if_str(part, image, "mediaType", "media_type")
        _copy_if_str(part, image, "data", "b64_json")
        _copy_if_str(part, image, "url", "url")
        return [cast(ResponseContentPart, image)]
    if part_type in {"output_file", "file"}:
        file_part: dict[str, Any] = {
            "type": "file",
            "providerMetadata": {"openai": dict(part)},
        }
        _copy_if_str(part, file_part, "mediaType", "media_type")
        _copy_if_str(part, file_part, "data", "file_data")
        _copy_if_str(part, file_part, "url", "file_url")
        _copy_if_str(part, file_part, "filename", "filename")
        return [cast(ResponseContentPart, file_part)]
    return []


def _collect_openai_responses_logprobs(
    response_json: Mapping[str, Any],
) -> Sequence[Any]:
    output = response_json.get("output")
    if not isinstance(output, Sequence):
        return ()
    logprobs: list[Any] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, Sequence):
            continue
        for part in content:
            if isinstance(part, Mapping) and "logprobs" in part:
                logprobs.append(part["logprobs"])
    return logprobs


def _sequence_or_empty(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return (value,)
    return ()


__all__ = [
    "CHAT_PROVIDER_OPTIONS",
    "RESPONSES_PROVIDER_OPTIONS",
    "_OpenAIEndpointClient",
    "_OpenAIResponsesClient",
    "build_chat_payload",
    "build_responses_payload",
    "model_capabilities",
    "parse_chat_response",
    "parse_responses_response",
]
