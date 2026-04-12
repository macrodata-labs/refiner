---
title: "Inference"
description: "(V)LLM inference workflow in Refiner"
---

Models in the loop, are now part of almost every processing step in modern data curation workflows. Therefore refiner provides a strong support for such use-cases.

It supports two modes:
- `OpenAIEndpointProvider`: call a direct OpenAI-compatible HTTP endpoint (OpenAI/OpenRouter etc..)
- `VLLMProvider`: ask Refiner Cloud to start and manage a VLLM server for you, giving you dedicated endpoint to call. Compared to using inference provider, this allows you to heavily paralelize, unlike with providers where you quickly hit rate-limit. As of right now this mode is only supported when running in the cloud.

## Usage

### OpenAIEndpointProvider
Use `OpenAIEndpointProvider` if you want to use genereric inference provider.

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
    max_in_flight=64, # Adjust to prevent overwheling your inference provider
)
```

### Refiner managed VLLM runtime

Use `VLLMProvider` when launching on Refiner Cloud and you want the platform to start and manage a VLLM server for you:

```python
import refiner as mdr

provider = mdr.inference.VLLMProvider(
    model_name_or_path="google/gemma-4-26B-A4B-it",
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
        max_concurrent_requests=512
        # Your generate function can create multiple generaate requests at a time (e.g video with multiple frames), this ensures we run 256 rows at at a time, but in total we never have more than 512 concurrent requests
    ),
    max_in_flight=256, # Sets max in-flight rows
)
```

#### Cold-Starts
Because we spawn the VLLM on fresh hardware, the VLLM startup can take from 2-20 minutes (depending on model size) due to model complications, model weight downloads and torch initialization. To speed-up this process, we provide a torch compile cache as well as model weights by default for following models:
- `google/gemma-4-26B-A4B-it`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`

While you can use models beyond the selection above, the startup time will be affected by the absence of the caches (only the first-time).

#### Inference Hardware
As of now, all VLLM deployments are served from 1x H100, which puts restriction on model sizes. However this is subject to change to allow
higher variability and availability.

## Examples

- Direct endpoint example: [examples/inference_endpoint.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_endpoint.py)
- VLLM-backed example: [examples/inference_vllm.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_vllm.py)