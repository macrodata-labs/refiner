---
title: "Inference"
description: "(V)LLM inference workflow in Refiner"
---

Models in the loop, are now part of almost every processing step in modern data curation workflows. Therefore refiner provides a strong support for such use-cases.

It supports two modes:
- `OpenAIEndpointProvider`: call a direct OpenAI-compatible HTTP endpoint (OpenAI/OpenRouter etc..)
- `VLLMProvider`: ask Refiner Cloud to start and manage a VLLM server for you, giving you dedicated endpoint to call. This is typically considreably cheaper and stable ("no rate-limit"). <TODO add some calculation>. As of right now this mode is only supported when running in the cloud.

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
    ),
    max_in_flight=256, # Adjust to prevent overwheliming the endpoint. 256 is recommend.
)
```

This mode is cloud-only. You provide the model config, and Refiner Cloud handles the VLLM server lifecycle for the job.

### Supported models
- `Qwen/Qwen3-VL-30B-A3B-Instruct` 16384
- `Qwen/Qwen3-VL-8B-Instruct` 24576
- `google/gemma-4-26B-A4B-it` 32768


## Examples

- Direct endpoint example: [examples/inference_endpoint.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_endpoint.py)
- VLLM-backed example: [examples/inference_vllm.py](/Users/hynky/.codex/worktrees/ab9a/refiner/examples/inference_vllm.py)