---
title: "Inference"
description: "Run row-oriented async inference against an OpenAI-compatible endpoint"
---

Refiner inference is a row-oriented wrapper around `.map_async(...)` for direct HTTP generation.

The API is:

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

## Configuration

`mdr.inference.generate(...)` accepts:

- `fn`: async or sync user function with signature `fn(row, generate) -> dict[str, object] | Row`
- `provider`: an `OpenAIEndpointProvider`
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

Use `OpenAIEndpointProvider` to describe the target endpoint:

```python
endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
)
```

Refiner treats this as a direct OpenAI-compatible HTTP endpoint. It does not start or manage a model runtime itself.

## Internal Notes

The inference helper builds a simple OpenAI-compatible HTTP client, merges default request parameters, and enforces request concurrency with a semaphore. It does not use service specs, runtime bindings, or worker-side service registries.
