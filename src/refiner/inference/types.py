from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict

JSONValue: TypeAlias = Any
ProviderOptions: TypeAlias = Mapping[str, Mapping[str, JSONValue]]
DataContent: TypeAlias = str | bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    model_family: str | None = None
    images: bool | None = None
    audio: bool | None = None
    video: bool | None = None
    files: bool | None = None
    tools: bool | None = None
    structured_output: bool | None = None
    reasoning: bool | None = None
    generated_media: bool | None = None
    citations: bool | None = None
    max_output_tokens: int | None = None
    system_message_mode: Literal["system", "developer", "remove"] | None = None
    flex_processing: bool | None = None
    priority_processing: bool | None = None
    non_reasoning_parameters: bool | None = None
    adaptive_thinking: bool | None = None
    xhigh_reasoning_effort: bool | None = None
    known_model: bool | None = None


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


class CustomPart(_ProviderOptionsPart):
    type: Literal["custom"]
    provider: str
    data: Mapping[str, Any]


class ReasoningPart(_ProviderOptionsPart):
    type: Literal["reasoning"]
    text: str


class _ProviderMetadataContentPart(TypedDict, total=False):
    providerMetadata: Mapping[str, Any]


class TextContentPart(_ProviderMetadataContentPart):
    type: Literal["text"]
    text: str


class ReasoningContentPart(_ProviderMetadataContentPart):
    type: Literal["reasoning"]
    text: str


class SourceContentPart(TypedDict, total=False):
    type: Literal["source"]
    sourceType: Literal["url", "file", "document", "unknown"]
    id: str
    url: str
    title: str
    providerMetadata: Mapping[str, Any]


class GeneratedFileContentPart(TypedDict, total=False):
    type: Literal["file"]
    mediaType: str
    data: str
    url: str
    filename: str
    providerMetadata: Mapping[str, Any]


class GeneratedImageContentPart(TypedDict, total=False):
    type: Literal["image"]
    mediaType: str
    data: str
    url: str
    providerMetadata: Mapping[str, Any]


class InferenceWarning(TypedDict, total=False):
    type: Literal[
        "unsupported-setting",
        "unsupported-provider-option",
        "unsupported-content",
        "other",
    ]
    message: str
    setting: str
    details: str


ProviderMetadata: TypeAlias = Mapping[str, Mapping[str, Any]]
ResponseContentPart: TypeAlias = (
    TextContentPart
    | ReasoningContentPart
    | SourceContentPart
    | GeneratedFileContentPart
    | GeneratedImageContentPart
)
UserContent: TypeAlias = str | Sequence[TextPart | ImagePart | FilePart | CustomPart]
AssistantContent: TypeAlias = (
    str | Sequence[TextPart | FilePart | ReasoningPart | CustomPart]
)


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: UserContent


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: AssistantContent


Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage


class OpenAIProviderOptions(TypedDict, total=False):
    audio: Mapping[str, Any]
    background: bool
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
    imageDetail: Literal["auto", "low", "high"]
    instructions: str | None
    logitBias: Mapping[int | str, int | float]
    logprobs: bool | int
    maxToolCalls: int | None
    modalities: Sequence[Literal["text", "audio"]]
    parallelToolCalls: bool
    previousResponseId: str | None
    responseFormat: Mapping[str, Any]
    text: Mapping[str, Any]
    topLogprobs: int
    user: str
    webSearchOptions: Mapping[str, Any]
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
    apiClient: str
    responseModalities: Sequence[Literal["TEXT", "IMAGE"]]
    responseMimeType: str
    responseSchema: Mapping[str, Any]
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
    speechConfig: Mapping[str, Any]
    systemInstruction: Mapping[str, Any]
    streamFunctionCallArguments: bool
    serviceTier: Literal["standard", "flex", "priority"]
    sharedRequestType: Literal["priority", "flex", "standard"]
    requestType: Literal["shared"]
    thoughtSignature: str


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
    serviceTier: Literal["auto", "standard_only"]
    toolStreaming: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    taskBudget: Mapping[str, Any]
    speed: Literal["fast", "standard"]
    inferenceGeo: Literal["us", "global"]
    anthropicBeta: Sequence[str]
    contextManagement: Mapping[str, Any]
    signature: str
    redactedData: str


class AnthropicCitations(TypedDict):
    enabled: bool


class AnthropicFilePartProviderOptions(TypedDict, total=False):
    citations: AnthropicCitations
    title: str
    context: str
    cacheControl: AnthropicCacheControl


__all__ = [
    "AnthropicCacheControl",
    "AnthropicCitations",
    "AnthropicFilePartProviderOptions",
    "AnthropicProviderOptions",
    "AnthropicThinking",
    "AssistantMessage",
    "AssistantContent",
    "DataContent",
    "CustomPart",
    "FilePart",
    "GeneratedFileContentPart",
    "GeneratedImageContentPart",
    "GoogleImageConfig",
    "GoogleProviderOptions",
    "GoogleSafetySetting",
    "GoogleThinkingConfig",
    "ImagePart",
    "InferenceWarning",
    "Message",
    "ModelCapabilities",
    "OpenAIProviderOptions",
    "ProviderMetadata",
    "ProviderOptions",
    "ReasoningContentPart",
    "ReasoningPart",
    "ResponseContentPart",
    "SourceContentPart",
    "SystemMessage",
    "TextPart",
    "TextContentPart",
    "UserContent",
    "UserMessage",
]
