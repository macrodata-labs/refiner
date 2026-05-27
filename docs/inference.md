---
title: "Inference"
description: "(V)LLM inference workflow in Refiner"
---

Model calls are now part of many data curation workflows, so Refiner includes built-in support for endpoint-based and managed inference.

It supports these modes:
- `OpenAIEndpointProvider`: call any OpenAI-compatible HTTP endpoint such as OpenAI or OpenRouter.
- `OpenAIResponsesProvider`: call OpenAI's native Responses API with the same
  `generate_text` message format.
- `GoogleEndpointProvider`: call the native Google Gemini API with Refiner's
  typed `generate_text` message format.
- `AnthropicEndpointProvider`: call Anthropic's native Messages API with the
  same typed `generate_text` message format.
- `VLLMProvider`: ask Refiner Cloud to start and manage a dedicated VLLM server for your job. This avoids external rate limits and is only supported when running on Refiner Cloud.

## Usage

### Typed text generation
Use `generate_text` for Vercel-style typed chat messages. It accepts plain
Python dictionaries with typed content parts, so multimodal requests stay
type-checkable without helper constructors.

```python
import refiner as mdr

endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    model="gpt-5-mini",
)

async def caption(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "file",
                        "mediaType": "image/png",
                        "data": row["image"],
                    },
                ],
            }
        ],
        temperature=0,
    )
    return {"caption": response.text}


pipeline = mdr.read_jsonl("images.jsonl").map_async(
    mdr.inference.generate_text(
        fn=caption,
        provider=endpoint,
        default_generation_params={"max_tokens": 256},
    ),
    max_in_flight=64,
)
```

`generate_text` exports `TypedDict` types such as `Message`, `UserMessage`,
`TextPart`, `ImagePart`, and `FilePart` for static checking. Refiner converts
those canonical content parts into the selected provider's wire format before
the request is sent. For example, image file parts become OpenAI `image_url`
parts for OpenAI-compatible endpoints, `input_file` parts for
`OpenAIResponsesProvider`, Gemini `inlineData` parts for
`GoogleEndpointProvider`, and Anthropic image/document blocks for
`AnthropicEndpointProvider`.

For byte-like media, `mediaType` may be either a full MIME type
(`"image/png"`, `"video/mp4"`) or a top-level media type/wildcard
(`"image"`, `"image/*"`, `"video"`). Refiner detects common byte signatures
such as PNG, JPEG, PDF, MP4, WAV, MP3, GIF, and WebP before sending the
provider request.

For single-turn text-only calls, pass `prompt` instead of `messages`:

```python
async def summarize(row, generate_text):
    response = await generate_text(prompt=f"Summarize this: {row['text']}")
    return {"summary": response.text}
```

`InferenceResponse.text` contains the concatenated text output.
`InferenceResponse.content` contains normalized rich parts:
`{"type": "text", "text": ...}`, `{"type": "reasoning", "text": ...}`,
`{"type": "source", ...}`, `{"type": "image", ...}`, and
`{"type": "file", ...}` when the provider returns citations, generated media, or
files. `InferenceResponse.logprobs` contains provider logprob payloads when
requested and returned. `InferenceResponse.provider_metadata` contains stable
provider metadata such as response IDs, model IDs, selected candidate metadata,
and safety/finish details. `InferenceResponse.headers` contains normalized
response headers from the provider, such as request IDs, retry headers, and
rate-limit metadata when the provider sends them.
`InferenceResponse.warnings` contains structured warnings for requests that can
still be sent but include options Refiner does not map for the selected
provider, such as unrelated provider option namespaces, unsupported
provider-specific settings, content that the selected model is not known to
support, large inline media, or tool options passed to `generate_text` before
tool calling is implemented. Refiner also uses provider/model capability tables
based on AI SDK behavior for known model families, including OpenAI reasoning
model parameter limits, OpenAI service-tier support, Anthropic adaptive thinking
support, Anthropic `xhigh` effort support, and Anthropic output-token limits.

```python
async def summarize(row, generate_text):
    response = await generate_text(
        prompt=row["text"],
        providerOptions={"google": {"thinkingConfig": {"thinkingBudget": 128}}},
    )
    return {"summary": response.text, "warnings": list(response.warnings)}
```

Assistant history can be passed as either plain text or typed content parts.
Google supports assistant text, reasoning, and in-memory file parts. Anthropic
supports assistant text and reasoning. OpenAI Responses supports assistant text
and reasoning. OpenAI-compatible chat endpoints support assistant text history.

### Retries and errors
Provider calls use the same retry shape as the Vercel AI SDK. Refiner retries
retryable failures up to `maxRetries=2` by default, with exponential backoff
starting at two seconds. Retryable failures are request timeout (`408`),
conflict (`409`), rate limiting (`429`), server errors (`>=500`), and network
connection failures. Refiner respects provider `retry-after-ms` and
`retry-after` headers when the delay is valid and less than 60 seconds.

Disable retries or override the retry count per call:

```python
async def summarize(row, generate_text):
    response = await generate_text(
        prompt=row["text"],
        maxRetries=0,
    )
    return {"summary": response.text}
```

Failed provider responses raise `mdr.inference.InferenceAPICallError`. The
exception includes `status_code`, `response_headers`, `response_body`, parsed
provider error `data`, the request `url`, and `is_retryable`. If retries are
exhausted, Refiner raises `mdr.inference.InferenceRetryError` with the collected
underlying errors and a reason such as `"maxRetriesExceeded"`.

### Structured output
Pass a Pydantic `BaseModel` class with `schema=` to request and validate
structured JSON output. Refiner sends provider-native schema hints for
OpenAI-compatible chat endpoints, OpenAI Responses, and Google Gemini. Anthropic
requests include an explicit JSON instruction and are validated locally, with a
warning because schema enforcement is not native.

```python
from pydantic import BaseModel


class Caption(BaseModel):
    title: str
    objects: list[str]


async def caption(row, generate_text):
    response = await generate_text(
        prompt="Describe this image as structured JSON.",
        schema=Caption,
    )
    caption = response.object
    return {
        "title": caption.title,
        "objects": caption.objects,
    }
```

If the response cannot be parsed as the requested Pydantic model, Refiner raises
`mdr.inference.InferenceSchemaValidationError`.

### GoogleEndpointProvider
Use `GoogleEndpointProvider` for native Gemini requests, including in-memory
video inputs. Set `GOOGLE_GENERATIVE_AI_API_KEY` in the worker environment.

```python
import refiner as mdr

provider = mdr.inference.GoogleEndpointProvider(
    model="gemini-2.5-flash",
)

async def summarize_video(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this video."},
                    {
                        "type": "file",
                        "mediaType": "video/mp4",
                        "data": row["video"],
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    return {"summary": response.text}


pipeline = mdr.read_jsonl("videos.jsonl").map_async(
    mdr.inference.generate_text(
        fn=summarize_video,
        provider=provider,
    ),
    max_in_flight=16,
)
```

Pass provider-specific options with Vercel-style namespaced `providerOptions`.
Refiner places body-level options on the correct provider request object and
part-level options on the relevant content part.

```python
async def analyze(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this video."},
                    {
                        "type": "file",
                        "mediaType": "video/mp4",
                        "data": row["video"],
                    },
                ],
            }
        ],
        providerOptions={
            "google": {
                "thinkingConfig": {"thinkingBudget": 128},
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    }
                ],
            }
        },
    )
    return {"summary": response.text}
```

Use a `custom` part as a provider-specific escape hatch when a provider exposes
a content block that Refiner does not normalize yet. The `provider` value must
match the selected adapter (`"openai"`, `"openai-responses"`, `"openai-chat"`,
`"google"`, or `"anthropic"`), and `data` is inserted into that provider's
request content as-is.

```python
async def analyze(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Use this provider-native input."},
                    {
                        "type": "custom",
                        "provider": "openai-responses",
                        "data": {"type": "input_image", "image_url": "file_abc"},
                    },
                ],
            }
        ]
    )
    return {"summary": response.text}
```

### OpenAIResponsesProvider
Use `OpenAIResponsesProvider` when you want OpenAI's native Responses API
instead of the chat-completions-compatible endpoint. Set `OPENAI_API_KEY` in the
worker environment.

```python
provider = mdr.inference.OpenAIResponsesProvider(model="gpt-5-mini")

async def summarize_pdf(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this PDF."},
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "data": row["pdf"],
                    },
                ],
            }
        ],
        providerOptions={
            "openai": {
                "reasoningEffort": "low",
                "textVerbosity": "low",
            }
        },
        max_tokens=512,
    )
    return {"summary": response.text}
```

### AnthropicEndpointProvider
Use `AnthropicEndpointProvider` for Anthropic's native Messages API. Set
`ANTHROPIC_API_KEY` in the worker environment.

```python
provider = mdr.inference.AnthropicEndpointProvider(
    model="claude-sonnet-4-5",
)

async def summarize_document(row, generate_text):
    response = await generate_text(
        messages=[
            {"role": "system", "content": "Use citations where possible."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document."},
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "filename": "paper.pdf",
                        "data": row["pdf"],
                        "providerOptions": {
                            "anthropic": {
                                "title": "Paper",
                                "citations": {"enabled": True},
                            }
                        },
                    },
                ],
            },
        ],
        providerOptions={
            "anthropic": {
                "thinking": {"type": "enabled", "budgetTokens": 1024}
            }
        },
        max_tokens=512,
    )
    return {"summary": response.text}
```

### OpenAIEndpointProvider
Use `OpenAIEndpointProvider` when you already have an OpenAI-compatible endpoint.

```python
import refiner as mdr

endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    model="gpt-5-mini",
)

async def summarize(row, generate):
    response = await generate(
        {
            "messages": [
                {"role": "system", "content": "Summarize the input briefly."},
                {"role": "user", "content": row["text"]},
            ]
        }
    )
    return {"summary": response.text}


pipeline = mdr.read_jsonl("input.jsonl").map_async(
    mdr.inference.generate(
        fn=summarize,
        provider=endpoint,
        default_generation_params={"temperature": 0.1, "max_tokens": 256},
    ),
    max_in_flight=64,  # Adjust to avoid overwhelming your endpoint
)
```

Set `OPENAI_API_KEY` in the worker environment before execution. For cloud jobs,
pass the key through `secrets={"OPENAI_API_KEY": None}` so it is available in
the worker environment without serializing the secret into the provider config.

### Refiner managed VLLM runtime

Use `VLLMProvider` when launching on Refiner Cloud and you want the platform to start and manage a VLLM server for you.
Cloud launches include the VLLM runtime service requirements in the submitted
stage plan and worker payload so the platform can reserve service capacity
before workers begin processing shards.

```python
import refiner as mdr

provider = mdr.inference.VLLMProvider(
    model="google/gemma-4-E4B-it",
    config="correctness",
)

async def summarize(row, generate):
    response = await generate(
        {
            "messages": [
                {"role": "system", "content": "Summarize the input briefly."},
                {"role": "user", "content": row["text"]},
            ]
        }
    )
    return {"summary": response.text}


pipeline = mdr.read_jsonl("input.jsonl").map_async(
    mdr.inference.generate(
        fn=summarize,
        provider=provider,
        default_generation_params={"temperature": 0.1, "max_tokens": 256},
    ),
    max_in_flight=256,
)
```

Use `config="throughput"` when you want the managed VLLM service to prioritize
serving throughput over the default correctness-oriented profile.

#### Supported models
Only the following models are currently supported. If you are missing one, please create an issue:
- `Qwen/Qwen3.5-9B`
- `google/gemma-4-E4B-it`

## Internal Notes

Inference endpoint clients share a transport layer that normalizes provider
HTTP errors, retry classification, retry-after handling, and response headers.
The total number of concurrent row-level requests is still controlled by
Refiner's execution layer through `max_concurrent_requests` and pipeline
`max_in_flight`, not by the single-call transport helper.

## Examples

- Direct endpoint example: [`examples/inference_endpoint.py`](../examples/inference_endpoint.py)
- VLLM-backed example: [`examples/inference_vllm.py`](../examples/inference_vllm.py)
