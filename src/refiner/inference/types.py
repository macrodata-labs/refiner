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
    logitBias: Mapping[int | str, int | float]
    logprobs: bool | int
    parallelToolCalls: bool
    user: str
    reasoningEffort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    maxCompletionTokens: int | float
    store: bool
    metadata: Mapping[str, Any]
    prediction: Mapping[str, Any]
    serviceTier: Literal["auto", "flex", "priority", "default"]
    strictJsonSchema: bool
    reasoningSummary: str
    textVerbosity: Literal["low", "medium", "high"]
    promptCacheKey: str
    promptCacheRetention: Literal["in_memory", "24h"]
    safetyIdentifier: str
    systemMessageMode: Literal["system", "developer", "remove"]
    forceReasoning: bool
    conversation: str | None
    include: (
        Sequence[
            Literal[
                "reasoning.encrypted_content",
                "file_search_call.results",
                "message.output_text.logprobs",
            ]
        ]
        | None
    )
    instructions: str | None
    maxToolCalls: int | None
    previousResponseId: str | None
    passThroughUnsupportedFiles: bool
    truncation: Literal["auto", "disabled"] | None
    contextManagement: Sequence[Mapping[str, Any]] | None
    allowedTools: Mapping[str, Any]


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
    structuredOutputs: bool
    safetySettings: Sequence[GoogleSafetySetting]
    threshold: Literal[
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE",
        "OFF",
    ]
    audioTimestamp: bool
    labels: Mapping[str, str]
    mediaResolution: Literal[
        "MEDIA_RESOLUTION_UNSPECIFIED",
        "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_HIGH",
    ]
    imageConfig: GoogleImageConfig
    retrievalConfig: Mapping[str, Any]
    streamFunctionCallArguments: bool
    serviceTier: Literal["standard", "flex", "priority"]
    sharedRequestType: Literal["priority", "flex", "standard"]
    requestType: Literal["shared"]


class AnthropicCacheControl(TypedDict, total=False):
    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"]


class AnthropicThinking(TypedDict, total=False):
    type: Literal["adaptive", "enabled", "disabled"]
    display: Literal["omitted", "summarized"]
    budgetTokens: int | float


class AnthropicProviderOptions(TypedDict, total=False):
    sendReasoning: bool
    structuredOutputMode: Literal["outputFormat", "jsonTool", "auto"]
    thinking: AnthropicThinking
    disableParallelToolUse: bool
    cacheControl: AnthropicCacheControl
    metadata: Mapping[str, Any]
    mcpServers: Sequence[Mapping[str, Any]]
    container: Mapping[str, Any]
    toolStreaming: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    taskBudget: Mapping[str, Any]
    speed: Literal["fast", "standard"]
    inferenceGeo: Literal["us", "global"]
    anthropicBeta: Sequence[str]
    contextManagement: Mapping[str, Any]


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ReasoningContentPart(TypedDict):
    type: Literal["reasoning"]
    text: str


ResponseContentPart: TypeAlias = TextContentPart | ReasoningContentPart


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
    "ReasoningContentPart",
    "ResponseContentPart",
    "SystemMessage",
    "TextPart",
    "TextContentPart",
    "UserContent",
    "UserMessage",
]
