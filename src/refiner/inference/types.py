from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypedDict

JSONValue: TypeAlias = Any
ProviderOptions: TypeAlias = Mapping[str, Mapping[str, JSONValue]]
DataContent: TypeAlias = str | bytes | bytearray | memoryview


class _ProviderOptionsPart(TypedDict, total=False):
    providerOptions: ProviderOptions


class _MediaTypePart(TypedDict, total=False):
    mediaType: str


class TextPart(_ProviderOptionsPart):
    type: Literal["text"]
    text: str


class ImagePart(_ProviderOptionsPart, _MediaTypePart):
    type: Literal["image"]
    image: DataContent


class FilePart(_ProviderOptionsPart):
    type: Literal["file"]
    data: DataContent
    mediaType: str


UserContent: TypeAlias = str | Sequence[TextPart | ImagePart | FilePart]


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: UserContent


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str


Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage


class OpenAIProviderOptions(TypedDict, total=False):
    imageDetail: Literal["auto", "low", "high"]
    reasoningEffort: Literal["minimal", "low", "medium", "high"]
    reasoningSummary: Literal["auto", "concise", "detailed"]
    textVerbosity: Literal["low", "medium", "high"]


class GoogleThinkingConfig(TypedDict, total=False):
    thinkingBudget: int | float
    includeThoughts: bool
    thinkingLevel: Literal["minimal", "low", "medium", "high"]


class GoogleSafetySetting(TypedDict):
    category: Literal[
        "HARM_CATEGORY_UNSPECIFIED",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    threshold: Literal[
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE",
        "OFF",
    ]


class GoogleImageConfig(TypedDict, total=False):
    aspectRatio: str
    imageSize: Literal["1K", "2K", "4K", "512"]


class GoogleProviderOptions(TypedDict, total=False):
    responseModalities: Sequence[Literal["TEXT", "IMAGE"]]
    thinkingConfig: GoogleThinkingConfig
    cachedContent: str
    safetySettings: Sequence[GoogleSafetySetting]
    audioTimestamp: bool
    labels: Mapping[str, str]
    mediaResolution: Literal[
        "MEDIA_RESOLUTION_UNSPECIFIED",
        "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_HIGH",
    ]
    imageConfig: GoogleImageConfig
    serviceTier: Literal["standard", "flex", "priority"]


class AnthropicCacheControl(TypedDict, total=False):
    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"]


class AnthropicThinking(TypedDict, total=False):
    type: Literal["adaptive", "enabled", "disabled"]
    display: Literal["omitted", "summarized"]
    budgetTokens: int | float


class AnthropicProviderOptions(TypedDict, total=False):
    thinking: AnthropicThinking
    cacheControl: AnthropicCacheControl
    metadata: Mapping[str, str]
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    speed: Literal["fast", "standard"]
    inferenceGeo: Literal["us", "global"]
    anthropicBeta: Sequence[str]


class AnthropicCitations(TypedDict):
    enabled: bool


class AnthropicFilePartProviderOptions(TypedDict, total=False):
    citations: AnthropicCitations
    title: str
    context: str


__all__ = [
    "AnthropicCacheControl",
    "AnthropicCitations",
    "AnthropicFilePartProviderOptions",
    "AnthropicProviderOptions",
    "AnthropicThinking",
    "AssistantMessage",
    "DataContent",
    "FilePart",
    "GoogleImageConfig",
    "GoogleProviderOptions",
    "GoogleSafetySetting",
    "GoogleThinkingConfig",
    "ImagePart",
    "Message",
    "OpenAIProviderOptions",
    "ProviderOptions",
    "SystemMessage",
    "TextPart",
    "UserContent",
    "UserMessage",
]
