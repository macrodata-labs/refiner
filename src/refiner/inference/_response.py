from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from refiner.inference.types import (
    InferenceWarning,
    ProviderMetadata,
    ResponseContentPart,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    text: str
    finish_reason: str | None
    usage: Mapping[str, Any]
    response: Mapping[str, Any]
    content: Sequence[ResponseContentPart] = ()
    headers: Mapping[str, str] = field(default_factory=dict)
    warnings: Sequence[InferenceWarning] = ()
    object: Any | None = None
    logprobs: Sequence[Any] = ()
    provider_metadata: ProviderMetadata = field(default_factory=dict)

    @property
    def raw(self) -> Mapping[str, Any]:
        return self.response


def _parse_inference_response(
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


def _parse_openai_responses_response(
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


def _parse_anthropic_inference_response(
    response_json: Mapping[str, Any],
    *,
    response_headers: Mapping[str, str] | None = None,
) -> InferenceResponse:
    content = response_json.get("content")
    if not isinstance(content, Sequence):
        raise RuntimeError("anthropic response is missing content")
    content_parts: list[ResponseContentPart] = []
    for part in content:
        if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
            continue
        part_type = part.get("type")
        if part_type == "text":
            content_parts.append({"type": "text", "text": part["text"]})
            content_parts.extend(_anthropic_sources(part.get("citations")))
        elif part_type in {"thinking", "reasoning"}:
            content_parts.append({"type": "reasoning", "text": part["text"]})
    text = _text_from_content(content_parts)
    if not text:
        raise RuntimeError("anthropic response is missing textual content")
    usage = response_json.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    mapped_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
    finish_reason = response_json.get("stop_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    return InferenceResponse(
        text=text,
        finish_reason=finish_reason,
        usage=mapped_usage,
        response=response_json,
        content=content_parts,
        headers=dict(response_headers or {}),
        provider_metadata=_provider_metadata("anthropic", response_json),
    )


def _parse_google_inference_response(
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


def _text_from_content(content: Sequence[ResponseContentPart]) -> str:
    return "".join(part["text"] for part in content if part["type"] == "text")


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


def _anthropic_sources(citations: object) -> list[ResponseContentPart]:
    if not isinstance(citations, Sequence) or isinstance(citations, str):
        return []
    sources: list[ResponseContentPart] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        citation = cast(Mapping[str, Any], citation)
        url = citation.get("url")
        title = citation.get("title") or citation.get("document_title")
        source: dict[str, Any] = {
            "type": "source",
            "sourceType": "url" if isinstance(url, str) else "document",
            "providerMetadata": {"anthropic": dict(citation)},
        }
        if isinstance(url, str):
            source["url"] = url
        if isinstance(title, str):
            source["title"] = title
        sources.append(cast(ResponseContentPart, source))
    return sources


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


def _provider_metadata(
    provider: str,
    response_json: Mapping[str, Any],
    selected: Mapping[str, Any] | None = None,
) -> ProviderMetadata:
    metadata: dict[str, Any] = {}
    for key in (
        "id",
        "model",
        "created",
        "metadata",
        "service_tier",
        "system_fingerprint",
    ):
        if key in response_json:
            metadata[key] = response_json[key]
    if selected is not None:
        for key in ("index", "finish_reason", "finishReason", "safetyRatings"):
            if key in selected:
                metadata[key] = selected[key]
    return {provider: metadata} if metadata else {}


def _copy_if_str(
    source: Mapping[str, Any],
    target: dict[str, Any],
    target_key: str,
    source_key: str,
) -> None:
    value = source.get(source_key)
    if isinstance(value, str):
        target[target_key] = value


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
    "InferenceResponse",
    "_parse_anthropic_inference_response",
    "_parse_google_inference_response",
    "_parse_inference_response",
    "_parse_openai_responses_response",
]
