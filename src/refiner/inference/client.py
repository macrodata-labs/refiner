from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    text: str
    finish_reason: str | None
    usage: Mapping[str, Any]
    response: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _OpenAIEndpointClient:
    base_url: str
    api_key: str | None = None
    headers: Mapping[str, str] | None = None

    async def generate(self, payload: Mapping[str, Any]) -> InferenceResponse:
        endpoint_path = (
            "/v1/chat/completions" if "messages" in payload else "/v1/completions"
        )
        headers = dict(self.headers or {})
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=60.0,
            headers=headers,
        ) as client:
            response = await client.post(endpoint_path, json=dict(payload))
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
            use_chat="messages" in payload,
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


__all__ = ["InferenceResponse"]
