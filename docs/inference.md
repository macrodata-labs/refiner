---
title: "Inference"
description: "Run row-oriented async inference with managed or direct LLM services"
---

Refiner inference is a row-oriented wrapper around `.map_async(...)` that works with service definitions.

The service-backed API is:

```python
import refiner as mdr

llm = mdr.services.llm(
    name="llm",
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    model_max_context=8192,
)

async def summarize(row, llm):
    response = await llm.generate(
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
        service_name="llm",
        fn=summarize,
        default_generation_params={"temperature": 0.1, "max_tokens": 256},
        max_in_flight=64,
    ),
    services=[llm],
)
```

`mdr.inference.generate(...)` returns an async row function. Refiner runs it through the normal async row executor, so one input row produces one output row patch.

## Configuration

`mdr.inference.generate(...)` accepts:

- `service_name`: the name of the service that your inference function should use
- `fn`: async or sync user function with signature `fn(row, service) -> dict[str, object] | Row`
- `default_generation_params`: default request payload fields merged into each generation call
- `max_in_flight`: total in-flight HTTP generation requests across all rows
- `max_concurrent_rows`: optional row-level concurrency cap; defaults to `max_in_flight`

Declare the actual service definitions on `.map_async(..., services=[...])`.

The `service` object passed into your function implements `generate(...)` and returns an `InferenceResponse` with:

- `text`
- `finish_reason`
- `usage`
- `response`

## Services

Managed service:

```python
llm = mdr.services.llm(
    name="llm",
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    model_max_context=8192,
)
```

This is a managed service definition. Refiner does not start the runtime itself. The executor must provide a resolved runtime binding for it at worker startup.

Direct endpoint service:

```python
import refiner as mdr

llm = mdr.services.llm_endpoint(
    name="llm",
    base_url="https://api.openai.com",
    api_key_env="OPENAI_API_KEY",
)
```

This is the direct “normal requests” path. It builds its client from the service definition alone and does not require executor-provided bindings.

## Bindings

Managed services are resolved through a bindings file passed to the worker entrypoint. The executor writes JSON like:

```json
{
  "services": [
    {
      "name": "llm",
      "kind": "llm",
      "endpoint": "http://127.0.0.1:9000/v1",
      "headers": {"Authorization": "Bearer ..."},
      "metadata": {}
    }
  ]
}
```

If a managed service such as `llm` is used without a matching binding, Refiner raises an explicit error during client construction.

## Internal Notes

Planning hoists unique service definitions onto each stage. Worker startup loads optional service bindings from `--service-bindings-path`, builds service clients, and exposes them through worker context. The inference helper only merges default request parameters and enforces request and row concurrency.
