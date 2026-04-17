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

### Refiner managed VLLM runtime

Use `VLLMProvider` when launching on Refiner Cloud and you want the platform to start and manage a VLLM server for you.

```python
import refiner as mdr

provider = mdr.inference.VLLMProvider(
    model="google/gemma-4-26B-A4B-it",
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

When a model needs extra vLLM serve flags, pass them through `extra_kwargs`. For example, some Qwen video setups require:

```python
provider = mdr.inference.VLLMProvider(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    extra_kwargs={"limit-mm-per-prompt": "video=1"},
)
```

#### Cold-Starts
Because Refiner Cloud may start VLLM on fresh hardware, startup can take 2 to 20 minutes depending on model size, weight downloads, and torch initialization. To reduce this, Refiner Cloud prewarms a small set of commonly used models:
- `google/gemma-4-26B-A4B-it`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`

The two Qwen vision-language models should currently be launched with a single worker because multi-worker cold starts can hit compilation race conditions.
Other models can still be used, but the first startup is usually slower.

#### Inference Hardware
At the moment, VLLM deployments run on `1x H100`, which limits the model sizes that fit. This may change as the cloud runtime expands.

## Examples

- Direct endpoint example: [`examples/inference_endpoint.py`](../examples/inference_endpoint.py)
- VLLM-backed example: [`examples/inference_vllm.py`](../examples/inference_vllm.py)
