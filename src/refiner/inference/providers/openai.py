from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from refiner.inference.internal.media import (
    base64_data,
    data_or_url,
    is_url,
    resolve_media_type,
    top_level_media_type,
)
from refiner.inference.internal.message_conversion import (
    _custom_provider_data,
    _provider_option,
)
from refiner.inference.internal.schema import StructuredOutputSchema
from refiner.inference.internal.response import (
    InferenceResponse,
    _copy_if_str,
    _provider_metadata,
    _text_from_content,
)
from refiner.inference.internal.transport import (
    AiohttpAPIClient,
    post_json_to_api,
    provider_request_options,
)
from refiner.inference.types import (
    InferenceWarning,
    Message,
    ModelCapabilities,
    ProviderOptions,
    ResponseContentPart,
)

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
    max_connections: int | None = None
    max_keepalive_connections: int | None = None
    _client: AiohttpAPIClient | None = field(default=None, init=False, repr=False)
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

    def _ensure_client(self) -> AiohttpAPIClient:
        client = self._client
        if client is None:
            client = AiohttpAPIClient(
                base_url=_normalize_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout_s=_ENDPOINT_TIMEOUT_SECONDS,
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            )
            self._client = client
        return client

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        use_chat = "messages" in payload
        endpoint_path = "v1/chat/completions" if use_chat else "v1/completions"
        request_payload, max_retries, extra_headers = provider_request_options(payload)
        if use_chat:
            _strip_unsupported_reasoning_settings(request_payload)
        api_response = await post_json_to_api(
            self._ensure_client(),
            endpoint_path,
            request_payload,
            operation="generation",
            max_retries=max_retries,
            extra_headers=extra_headers,
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
        request_payload, max_retries, extra_headers = provider_request_options(payload)
        api_response = await post_json_to_api(
            self._ensure_client(),
            "pooling",
            request_payload,
            operation="pooling",
            max_retries=max_retries,
            extra_headers=extra_headers,
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
    max_connections: int | None = None
    max_keepalive_connections: int | None = None
    _client: AiohttpAPIClient | None = field(default=None, init=False, repr=False)
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

    def _ensure_client(self) -> AiohttpAPIClient:
        client = self._client
        if client is None:
            client = AiohttpAPIClient(
                base_url=_normalize_base_url(self.base_url),
                headers=self._resolved_headers,
                timeout_s=_ENDPOINT_TIMEOUT_SECONDS,
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            )
            self._client = client
        return client

    async def generate_text(self, payload: Mapping[str, Any]) -> InferenceResponse:
        request_payload, max_retries, extra_headers = provider_request_options(payload)
        _strip_unsupported_reasoning_settings(request_payload)
        api_response = await post_json_to_api(
            self._ensure_client(),
            "v1/responses",
            request_payload,
            operation="openai responses generation",
            max_retries=max_retries,
            extra_headers=extra_headers,
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


def model_capabilities(model: str, *, responses_api: bool) -> ModelCapabilities:
    reasoning = (
        model.startswith(("o1", "o3", "o4-mini"))
        or model.startswith("gpt-5")
        and not model.startswith("gpt-5-chat")
    )
    flex_processing = (
        model.startswith(("o3", "o4-mini"))
        or model.startswith("gpt-5")
        and not model.startswith("gpt-5-chat")
    )
    priority_processing = (
        model.startswith("gpt-4")
        or model.startswith(("o3", "o4-mini"))
        or (
            model.startswith("gpt-5")
            and not model.startswith(("gpt-5-nano", "gpt-5-chat", "gpt-5.4-nano"))
        )
    )
    non_reasoning_parameters = model.startswith(
        ("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5")
    )
    vision = _openai_vision_model(model)
    return ModelCapabilities(
        model_family="openai",
        images=vision,
        audio="audio" in model or "realtime" in model,
        video=False,
        files=responses_api,
        tools=True,
        structured_output=True,
        reasoning=reasoning,
        generated_media="image" in model,
        citations=responses_api,
        system_message_mode="developer" if reasoning else "system",
        flex_processing=flex_processing,
        priority_processing=priority_processing,
        non_reasoning_parameters=non_reasoning_parameters,
        known_model=_openai_known_model(model),
    )


def model_setting_warnings(
    *,
    model: str,
    provider_name: str,
    responses_api: bool,
    params: Mapping[str, Any],
    provider_options: Mapping[str, Mapping[str, Any]] | None,
) -> list[InferenceWarning]:
    model = model.lower()
    capabilities = model_capabilities(model, responses_api=responses_api)
    options = _provider_options(provider_options, "openai")
    warnings: list[InferenceWarning] = []

    if options.get("reasoningEffort") is not None and capabilities.reasoning is False:
        warnings.append(
            _unsupported_setting(
                f"{provider_name} model {model!r} is not known to support "
                "reasoningEffort.",
                setting="providerOptions.openai.reasoningEffort",
                details="AI SDK only enables reasoning effort for known reasoning models.",
            )
        )
    if (
        responses_api
        and options.get("reasoningSummary") is not None
        and capabilities.reasoning is False
    ):
        warnings.append(
            _unsupported_setting(
                f"{provider_name} model {model!r} is not known to support "
                "reasoningSummary.",
                setting="providerOptions.openai.reasoningSummary",
                details="AI SDK only enables reasoning summaries for known reasoning models.",
            )
        )
    if not responses_api and options.get("reasoningSummary") is not None:
        warnings.append(
            _unsupported_setting(
                "OpenAI chat-completions provider options do not support "
                "reasoningSummary; use OpenAIResponsesProvider for reasoning summaries.",
                setting="providerOptions.openai.reasoningSummary",
            )
        )

    service_tier = options.get("serviceTier")
    if service_tier == "flex" and capabilities.flex_processing is False:
        warnings.append(
            _unsupported_setting(
                f"{provider_name} model {model!r} is not known to support "
                "flex service tier.",
                setting="providerOptions.openai.serviceTier",
                details="AI SDK enables flex processing for o3, o4-mini, and GPT-5 non-chat models.",
            )
        )
    if service_tier == "priority" and capabilities.priority_processing is False:
        warnings.append(
            _unsupported_setting(
                f"{provider_name} model {model!r} is not known to support "
                "priority service tier.",
                setting="providerOptions.openai.serviceTier",
                details=(
                    "AI SDK enables priority processing for GPT-4, selected GPT-5, "
                    "o3, and o4-mini models."
                ),
            )
        )

    reasoning_effort = options.get("reasoningEffort")
    allows_non_reasoning_params = (
        reasoning_effort == "none" and capabilities.non_reasoning_parameters is True
    )
    if capabilities.reasoning is True and not allows_non_reasoning_params:
        for setting, label in (
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("topP", "topP"),
            ("logprobs", "logprobs"),
            ("top_logprobs", "topLogprobs"),
            ("frequency_penalty", "frequencyPenalty"),
            ("presence_penalty", "presencePenalty"),
            ("logit_bias", "logitBias"),
        ):
            if setting in params or setting in options:
                warnings.append(
                    _unsupported_setting(
                        f"{provider_name} model {model!r} is a reasoning model; "
                        f"{label} may be rejected unless reasoningEffort is 'none' "
                        "on GPT-5.1+ models.",
                        setting=setting
                        if setting in params
                        else f"providerOptions.openai.{setting}",
                    )
                )
    return warnings


def _strip_unsupported_reasoning_settings(payload: dict[str, Any]) -> None:
    model = str(payload.get("model", "")).lower()
    responses_api = "input" in payload
    capabilities = model_capabilities(model, responses_api=responses_api)
    if capabilities.reasoning is not True:
        return
    reasoning_effort = payload.get("reasoning_effort")
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning_effort is None:
        reasoning_effort = reasoning.get("effort")
    supports_non_reasoning_parameters = (
        reasoning_effort == "none" and capabilities.non_reasoning_parameters is True
    )
    if not supports_non_reasoning_parameters:
        for key in ("temperature", "top_p", "logprobs"):
            payload.pop(key, None)
    for key in (
        "frequency_penalty",
        "presence_penalty",
        "top_logprobs",
        "logit_bias",
    ):
        payload.pop(key, None)
    if (
        not responses_api
        and "max_tokens" in payload
        and "max_completion_tokens" not in payload
    ):
        payload["max_completion_tokens"] = payload.pop("max_tokens")


def _openai_known_model(model: str) -> bool:
    return model.startswith(
        (
            "gpt-3.5",
            "gpt-4",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-5",
            "o1",
            "o3",
            "o4",
        )
    )


def _openai_vision_model(model: str) -> bool:
    return model.startswith(
        (
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-5",
            "o3",
            "o4",
        )
    )


def _provider_options(
    provider_options: Mapping[str, Mapping[str, Any]] | None,
    namespace: str,
) -> Mapping[str, Any]:
    if provider_options is None:
        return {}
    options = provider_options.get(namespace)
    return options if isinstance(options, Mapping) else {}


def _unsupported_setting(
    message: str,
    *,
    setting: str,
    details: str | None = None,
) -> InferenceWarning:
    warning: InferenceWarning = {
        "type": "unsupported-setting",
        "setting": setting,
        "message": message,
    }
    if details is not None:
        warning["details"] = details
    return warning


def build_chat_payload(
    *,
    messages: Sequence[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    payload["messages"] = convert_to_openai_chat_messages(messages)
    _normalize_reasoning_options(payload, provider_options)
    _normalize_text_options(payload, provider_options)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": _json_schema(schema),
        }
    return payload


def build_responses_payload(
    *,
    messages: Sequence[Message],
    params: Mapping[str, Any],
    provider_options: ProviderOptions | None,
    schema: StructuredOutputSchema | None,
) -> dict[str, Any]:
    payload = dict(params)
    payload["input"] = convert_to_openai_responses_input(messages)
    _apply_responses_options(payload, provider_options)
    _normalize_responses_params(payload)
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
        "metadata",
        "store",
        "truncation",
        "user",
        "text",
    }
    for key in passthrough:
        if key in openai_options:
            payload[key] = openai_options[key]
    _apply_aliases(
        payload,
        openai_options,
        {
            "maxToolCalls": "max_tool_calls",
            "maxCompletionTokens": "max_output_tokens",
            "parallelToolCalls": "parallel_tool_calls",
            "previousResponseId": "previous_response_id",
            "promptCacheKey": "prompt_cache_key",
            "promptCacheRetention": "prompt_cache_retention",
            "safetyIdentifier": "safety_identifier",
            "serviceTier": "service_tier",
            "topLogprobs": "top_logprobs",
            "contextManagement": "context_management",
        },
    )
    reasoning: dict[str, Any] = {}
    if "reasoningEffort" in openai_options:
        reasoning["effort"] = openai_options["reasoningEffort"]
    if "reasoningSummary" in openai_options:
        reasoning["summary"] = openai_options["reasoningSummary"]
    if reasoning:
        payload["reasoning"] = {**dict(payload.get("reasoning", {})), **reasoning}
    if "textVerbosity" in openai_options:
        text = dict(payload.get("text", {}))
        text["verbosity"] = openai_options["textVerbosity"]
        payload["text"] = text
    if "logprobs" in openai_options and "top_logprobs" not in payload:
        logprobs = openai_options["logprobs"]
        if isinstance(logprobs, bool):
            payload["top_logprobs"] = 20 if logprobs else None
        else:
            payload["top_logprobs"] = logprobs
        include = payload.get("include")
        include_values = (
            list(include)
            if isinstance(include, Sequence) and not isinstance(include, str)
            else []
        )
        if "message.output_text.logprobs" not in include_values:
            include_values.append("message.output_text.logprobs")
        payload["include"] = include_values


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
    for key in ("user", "modalities", "audio", "store", "metadata", "prediction"):
        if key in openai_options:
            payload[key] = openai_options[key]
    _apply_aliases(
        payload,
        openai_options,
        {
            "logitBias": "logit_bias",
            "parallelToolCalls": "parallel_tool_calls",
            "maxCompletionTokens": "max_completion_tokens",
            "serviceTier": "service_tier",
            "promptCacheKey": "prompt_cache_key",
            "promptCacheRetention": "prompt_cache_retention",
            "safetyIdentifier": "safety_identifier",
            "webSearchOptions": "web_search_options",
            "responseFormat": "response_format",
        },
    )
    if "logprobs" in openai_options:
        logprobs = openai_options["logprobs"]
        if isinstance(logprobs, bool):
            payload["logprobs"] = True if logprobs else None
            payload["top_logprobs"] = 0 if logprobs else None
        else:
            payload["logprobs"] = True
            payload["top_logprobs"] = logprobs
    if "topLogprobs" in openai_options:
        payload["top_logprobs"] = openai_options["topLogprobs"]
    reasoning: dict[str, Any] = {}
    if "reasoningEffort" in openai_options:
        payload["reasoning_effort"] = openai_options["reasoningEffort"]
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
        payload["verbosity"] = openai_options["textVerbosity"]


def _apply_aliases(
    payload: dict[str, Any],
    options: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> None:
    for source, target in aliases.items():
        if source in options:
            payload[target] = options[source]


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
            reasoning_part: dict[str, Any] = {"type": "reasoning", "text": reasoning}
            message_metadata = _openai_message_metadata(message)
            if message_metadata:
                reasoning_part["providerMetadata"] = {"openai": message_metadata}
            content_parts.append(cast(ResponseContentPart, reasoning_part))
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
        provider_metadata=_openai_chat_metadata(response_json, choice),
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
                                {
                                    "type": "reasoning",
                                    "text": summary_part["text"],
                                    "providerMetadata": {
                                        "openai": _openai_response_item_metadata(item),
                                    },
                                }
                            )
                continue
            if item_type == "image_generation_call":
                content_parts.append(_openai_image_generation_part(item))
                continue
            content = item.get("content")
            if not isinstance(content, Sequence):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if isinstance(part.get("text"), str):
                    content_parts.append(
                        {
                            "type": "text",
                            "text": part["text"],
                            "providerMetadata": {
                                "openai": _openai_response_item_metadata(part),
                            },
                        }
                    )
                    content_parts.extend(_openai_sources(part.get("annotations")))
                content_parts.extend(_openai_generated_parts(part))
    if not _text_from_content(content_parts) and isinstance(
        response_json.get("output_text"), str
    ):
        content_parts.append({"type": "text", "text": response_json["output_text"]})
    text = _text_from_content(content_parts)
    if not text and not content_parts:
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
        provider_metadata=_openai_responses_metadata(response_json),
    )


def _openai_message_metadata(message: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: message[key]
        for key in ("refusal", "audio", "annotations")
        if key in message
    }


def _openai_response_item_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item[key]
        for key in (
            "id",
            "type",
            "status",
            "role",
            "encrypted_content",
            "summary",
            "logprobs",
            "annotations",
        )
        if key in item
    }


def _openai_image_generation_part(item: Mapping[str, Any]) -> ResponseContentPart:
    result: dict[str, Any] = {
        "type": "image",
        "mediaType": _openai_image_generation_media_type(item),
        "providerMetadata": {"openai": _openai_response_item_metadata(item)},
    }
    image_data = item.get("result")
    if isinstance(image_data, str):
        result["data"] = image_data
    return cast(ResponseContentPart, result)


def _openai_image_generation_media_type(item: Mapping[str, Any]) -> str:
    output_format = item.get("output_format")
    if isinstance(output_format, str) and output_format:
        return f"image/{output_format.lower()}"
    return "image/png"


def _openai_chat_metadata(
    response_json: Mapping[str, Any],
    choice: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    metadata = dict(
        _provider_metadata("openai", response_json, choice).get("openai", {})
    )
    message = choice.get("message")
    if isinstance(message, Mapping):
        message_metadata = _openai_message_metadata(message)
        if message_metadata:
            metadata["message"] = message_metadata
    return {"openai": metadata} if metadata else {}


def _openai_responses_metadata(
    response_json: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    metadata = dict(_provider_metadata("openai", response_json).get("openai", {}))
    for key in (
        "object",
        "created_at",
        "status",
        "background",
        "error",
        "incomplete_details",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "store",
        "temperature",
        "text",
        "tool_choice",
        "top_logprobs",
        "top_p",
        "truncation",
        "usage",
        "user",
    ):
        if key in response_json:
            metadata[key] = response_json[key]
    return {"openai": metadata} if metadata else {}


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
    "model_setting_warnings",
    "parse_chat_response",
    "parse_responses_response",
]
