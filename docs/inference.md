---
title: "Inference"
description: "(V)LLM inference workflow in Refiner"
---

Refiner can call LLM providers from inside a pipeline map step. Use
`generate_text` for typed chat-style requests and `generate_pooling` for VLLM
pooling/scoring endpoints.

Supported providers:
- `OpenAIEndpointProvider`: OpenAI-compatible chat/completions endpoints.
- `OpenAIResponsesProvider`: OpenAI Responses API.
- `GoogleEndpointProvider`: Gemini API.
- `AnthropicEndpointProvider`: Anthropic Messages API.
- `VLLMProvider`: Refiner-managed VLLM on Refiner Cloud.

## Basic Usage

```python
import refiner as mdr

provider = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    model="gpt-5-mini",
)


async def summarize(row, generate_text):
    response = await generate_text(
        messages=[
            {"role": "user", "content": f"Summarize this: {row['text']}"}
        ],
        temperature=0.1,
        max_tokens=256,
    )
    return {"summary": response.text}


pipeline = mdr.read_jsonl("input.jsonl").map_async(
    mdr.inference.generate_text(
        fn=summarize,
        provider=provider,
    ),
    max_in_flight=64,
)
```

Set the provider API key in the worker environment:
- `OPENAI_API_KEY`
- `GOOGLE_GENERATIVE_AI_API_KEY`
- `ANTHROPIC_API_KEY`

## Multimodal Input

Messages can contain typed content parts. Refiner converts them to each
provider's request format.

```python
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
    )
    return {"caption": response.text}
```

For byte-like media, `mediaType` can be a full MIME type like `"image/png"` or
`"video/mp4"`, or a broad type like `"image"` / `"video"` when Refiner can
detect the format.

## Gemini Video

```python
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
```

## Structured Output

Pass a Pydantic model with `schema=` when you want validated JSON output.

```python
from pydantic import BaseModel


class Caption(BaseModel):
    title: str
    objects: list[str]


async def caption(row, generate_text):
    response = await generate_text(
        messages=[
            {"role": "user", "content": "Describe this image as JSON."}
        ],
        schema=Caption,
    )
    caption = response.object
    return {"title": caption.title, "objects": caption.objects}
```

If parsing or validation fails, Refiner raises
`mdr.inference.InferenceSchemaValidationError`.

## Provider Options

Use `providerOptions` for provider-specific knobs.

```python
async def analyze(row, generate_text):
    response = await generate_text(
        messages=[{"role": "user", "content": row["text"]}],
        providerOptions={
            "openai": {
                "reasoningEffort": "low",
                "textVerbosity": "low",
            }
        },
    )
    return {"answer": response.text}
```

```python
async def analyze(row, generate_text):
    response = await generate_text(
        messages=[{"role": "user", "content": row["text"]}],
        providerOptions={
            "google": {
                "thinkingConfig": {"thinkingBudget": 128}
            }
        },
    )
    return {"answer": response.text}
```

```python
async def analyze(row, generate_text):
    response = await generate_text(
        messages=[{"role": "user", "content": row["text"]}],
        providerOptions={
            "anthropic": {
                "thinking": {"type": "enabled", "budgetTokens": 1024}
            }
        },
    )
    return {"answer": response.text}
```

## Raw Payloads

Use `raw_payload` when you need to send the provider request body yourself.
Refiner still handles concurrency, retries, metrics, and response parsing.

```python
async def summarize(row, generate_text):
    response = await generate_text(
        raw_payload={
            "messages": [
                {"role": "system", "content": "Summarize briefly."},
                {"role": "user", "content": row["text"]},
            ]
        }
    )
    return {"summary": response.text}
```

## Responses

`response.text` is the concatenated text output.

Useful fields:
- `response.content`: normalized rich parts such as text, reasoning, sources,
  images, and files.
- `response.object`: parsed Pydantic object when `schema=` is used.
- `response.warnings`: non-fatal provider/model warnings.
- `response.headers`: provider response headers.
- `response.provider_metadata`: provider-specific response metadata.

## VLLM On Refiner Cloud

```python
provider = mdr.inference.VLLMProvider(
    model="google/gemma-4-E4B-it",
    config="correctness",
)


async def summarize(row, generate_text):
    response = await generate_text(
        messages=[{"role": "user", "content": row["text"]}],
        max_tokens=256,
    )
    return {"summary": response.text}
```

Use `config="throughput"` when serving speed matters more than the default
correctness-oriented profile.

## Examples

- Direct endpoint example: [`examples/inference_endpoint.py`](../examples/inference_endpoint.py)
- VLLM-backed example: [`examples/inference_vllm.py`](../examples/inference_vllm.py)
