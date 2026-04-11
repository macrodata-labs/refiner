---
title: "Inference"
description: "Run row-oriented async text generation against a direct endpoint or a cloud-managed VLLM runtime"
---

Refiner inference is a row-oriented wrapper around `.map_async(...)` for text generation.

It supports two provider modes:

- `OpenAIEndpointProvider`: call a direct OpenAI-compatible HTTP endpoint
- `VLLMProvider`: ask Refiner Cloud to start and manage a VLLM server for the model you specify

## Direct endpoint

Use `OpenAIEndpointProvider` when you already have an OpenAI-compatible HTTP endpoint:

```python
import refiner as mdr

endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
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
    max_in_flight=64,
)
```

`mdr.inference.generate(...)` returns an async row function. Refiner runs it through the normal async row executor, so one input row produces one output row patch.

## VLLM-backed runtime

Use `VLLMProvider` when launching on Refiner Cloud and you want the platform to start and manage a VLLM server for you:

```python
import refiner as mdr

provider = mdr.inference.VLLMProvider(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    model_max_context=8192,
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
    max_in_flight=64,
)
```

This mode is cloud-only. You provide the model config, and Refiner Cloud handles the VLLM server lifecycle for the job.

## Configuration

`mdr.inference.generate(...)` accepts:

- `fn`: async or sync user function with signature `fn(row, generate) -> dict[str, object] | Row`
- `provider`: either an `OpenAIEndpointProvider` or a `VLLMProvider`
- `default_generation_params`: default request payload fields merged into each generation call

`.map_async(...)` still controls async row execution:

- `max_in_flight`: maximum number of rows processed concurrently
- `preserve_order`: whether results are emitted in input order

The `generate` callback passed into your function returns an `InferenceResponse` with:

- `text`
- `finish_reason`
- `usage`
- `response`

This helper is currently text-generation-oriented. It expects text completions or chat completions and returns `response.text`. It is not the right API yet for embeddings or image-generation-style outputs.

## Provider

### `OpenAIEndpointProvider`

Use `OpenAIEndpointProvider` to describe a direct target endpoint:

```python
endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
)
```

Refiner treats this as a direct OpenAI-compatible HTTP endpoint. It does not start or manage a model runtime itself.

### `VLLMProvider`

Use `VLLMProvider` to describe the model that should be served by the executor-managed runtime:

```python
provider = mdr.inference.VLLMProvider(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    model_max_context=8192,
)
```

This provider only carries model configuration. It does not contain a base URL, and it does not start a local runtime. When you launch on cloud, Refiner Cloud starts and manages the matching VLLM server for the job.

## Examples

- Direct endpoint example: [examples/inference_endpoint.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_endpoint.py)
- VLLM-backed example: [examples/inference_vllm.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_vllm.py)

The endpoint example runs against an OpenAI-compatible HTTP endpoint.
The VLLM example is meant for cloud execution, where Refiner Cloud manages the server for you.

## Internal Notes

Refiner sends OpenAI-compatible requests and merges `default_generation_params` into each call. With `OpenAIEndpointProvider`, requests go directly to the configured endpoint. With `VLLMProvider` on cloud, Refiner Cloud manages the server and the step uses that managed endpoint automatically.
