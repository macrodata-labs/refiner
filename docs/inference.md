---
title: "Inference"
description: "Run row-oriented async inference against a direct endpoint or a VLLM-backed runtime"
---

Refiner inference is a row-oriented wrapper around `.map_async(...)` for model generation.

It supports two provider modes:

- `OpenAIEndpointProvider`: call a direct OpenAI-compatible HTTP endpoint
- `VLLMProvider`: declare a VLLM-backed runtime requirement and resolve the actual endpoint from executor-provided runtime bindings

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
        max_concurrent_requests=64,
    ),
    max_in_flight=64,
)
```

`mdr.inference.generate(...)` returns an async row function. Refiner runs it through the normal async row executor, so one input row produces one output row patch.

## VLLM-backed runtime

Use `VLLMProvider` when the executor is responsible for provisioning a runtime service and handing workers a resolved endpoint at execution time:

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
        max_concurrent_requests=64,
    ),
    max_in_flight=64,
)
```

This provider does not contain an endpoint directly. During planning it emits a runtime service spec, and during execution it expects the worker runtime to provide a matching `VLLMRuntimeServiceBinding`.

## Configuration

`mdr.inference.generate(...)` accepts:

- `fn`: async or sync user function with signature `fn(row, generate) -> dict[str, object] | Row`
- `provider`: either an `OpenAIEndpointProvider` or a `VLLMProvider`
- `default_generation_params`: default request payload fields merged into each generation call
- `max_concurrent_requests`: total in-flight HTTP generation requests across all rows

`.map_async(...)` still controls async row execution:

- `max_in_flight`: maximum number of rows processed concurrently
- `preserve_order`: whether results are emitted in input order

The `generate` callback passed into your function returns an `InferenceResponse` with:

- `text`
- `finish_reason`
- `usage`
- `response`

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

This provider only carries model configuration. It does not contain a base URL or start a local runtime. Instead:

- planning converts it into a `VLLMServiceDefinition`
- the service definition emits a runtime service spec
- the executor provisions the runtime
- workers resolve the matching `VLLMRuntimeServiceBinding` by service name at execution time

## Examples

- Direct endpoint example: [examples/inference_endpoint.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_endpoint.py)
- VLLM-backed example: [examples/inference_vllm.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_vllm.py)

The endpoint example runs locally against an OpenAI-compatible HTTP endpoint.
The VLLM example declares a managed runtime requirement and is meant for cloud
execution where the executor provisions the matching runtime service binding.

## Internal Notes

The inference helper builds an OpenAI-compatible HTTP client, merges default request parameters, and enforces request concurrency with a semaphore. Endpoint-backed inference uses only the provider config. VLLM-backed inference emits a runtime service spec during planning and resolves a `VLLMRuntimeServiceBinding` from worker context at execution time.
