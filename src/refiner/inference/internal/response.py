from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from refiner.inference.types import (
    InferenceWarning,
    ProviderMetadata,
    ResponseContentPart,
)


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


def _text_from_content(content: Sequence[ResponseContentPart]) -> str:
    return "".join(part["text"] for part in content if part["type"] == "text")


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


__all__ = ["InferenceResponse"]
