---
title: "Inference"
description: "(V)LLM inference workflow in Refiner"
---

Model calls are now part of many data curation workflows, so Refiner includes built-in support for endpoint-based and managed inference.

It supports two modes:
- `OpenAIEndpointProvider`: call any OpenAI-compatible HTTP endpoint such as OpenAI or OpenRouter.
- `VLLMProvider`: ask Refiner Cloud to start and manage a dedicated VLLM server for your job. This avoids external rate limits and is only supported when running on Refiner Cloud.

## Usage

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

Set `OPENAI_API_KEY` in the worker environment before execution. For cloud jobs, pass it through `secrets={"OPENAI_API_KEY": None}`.

### Request limits

Refiner uses adaptive request limiting by default. It starts below the configured
maximum, grows after clean success windows, and backs off when an endpoint
returns HTTP `429`.

```python
import refiner as mdr

endpoint = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    model="gpt-5-mini",
)

pipeline = mdr.read_jsonl("input.jsonl").map_async(
    mdr.inference.generate(
        fn=summarize,
        provider=endpoint,
        rate_limit=mdr.inference.AdaptiveRateLimit(
            max_concurrency=256,
            initial_concurrency=16,
        ),
    ),
    max_in_flight=256,
)
```

When `rate_limit` is omitted or set to `None`, Refiner uses
`AdaptiveRateLimit()` with its defaults. Growth uses a shrinking multiplier: the
default starts at `2.0`, subtracts `0.1` after each successful growth window,
and never drops below `1.05`. A rate-limit response halves the current request
concurrency by default and subtracts `0.2` from the growth multiplier. If the
endpoint sends `Retry-After`, Refiner waits before starting more requests.

Use `StaticRateLimit` when you want the old fixed-concurrency behavior:

```python
pipeline = mdr.read_jsonl("input.jsonl").map_async(
    mdr.inference.generate(
        fn=summarize,
        provider=endpoint,
        rate_limit=mdr.inference.StaticRateLimit(max_concurrency=64),
    ),
    max_in_flight=64,
)
```

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


## Examples

- Direct endpoint example: [`examples/inference_endpoint.py`](../examples/inference_endpoint.py)
- VLLM-backed example: [`examples/inference_vllm.py`](../examples/inference_vllm.py)

## Internal Notes

Adaptive request limiting is process-local. In multi-worker jobs, each worker
adapts independently to the same provider. This reduces per-worker bursts but is
not a global quota coordinator across the full job.
