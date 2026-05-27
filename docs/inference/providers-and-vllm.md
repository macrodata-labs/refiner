---
title: "Providers and VLLM"
description: "Configure model providers for Refiner inference"
---

# Providers And VLLM

Providers describe where inference requests go and how payloads are formatted.

## OpenAI Responses

```python
provider = mdr.inference.OpenAIResponsesProvider(
    model="gpt-4.1",
)
```

## OpenAI-Compatible Endpoint

```python
provider = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.example.com/v1",
    model="gpt-4.1",
)
```

## Google Endpoint

```python
provider = mdr.inference.GoogleEndpointProvider(
    model="gemini-2.5-pro",
)
```

## Anthropic Endpoint

```python
provider = mdr.inference.AnthropicEndpointProvider(
    model="claude-sonnet-4",
)
```

## VLLM

```python
provider = mdr.inference.VLLMProvider(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
)
```

When used with cloud launch, VLLM providers can be represented as runtime
services so workers call a model server instead of loading the model directly
inside every worker.

## Provider Options

Provider-specific options are passed through `providerOptions`:

Provider options are typed objects for model-specific request settings. They are
passed as `providerOptions` to `generate_text(...)`.

Use provider options sparingly. Keep pipeline code portable unless you need a
specific model feature.

## Related Pages

- [Generate Text](generate-text.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
