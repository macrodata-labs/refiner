---
title: "Inference providers"
description: "Configure model providers for Refiner inference"
---

# Inference providers

Providers define where inference requests are sent and how Refiner formats each
request for that backend. Use them with `generate_text(...)` and other inference
helpers when you want the same pipeline code to run against hosted APIs or managed inference service.

Refiner supports OpenAI, OpenAI-compatible endpoints, Google Gemini, Anthropic
Claude, and managed vLLM services.

## OpenAI responses

Use `OpenAIResponsesProvider` for OpenAI-hosted models through the Responses
API. Choose it for OpenAI multimodal, tool-use, or structured-output workloads
because it maps to OpenAI's current inference surface.

```python
provider = mdr.inference.OpenAIResponsesProvider(
    model="gpt-4.1",
)
```

## OpenAI-compatible endpoint

Use `OpenAIEndpointProvider` for services that expose an OpenAI-compatible
`/v1/chat/completions` API but are not OpenAI. Choose it for hosted gateways,
local model servers, and third-party providers that implement the OpenAI chat
API shape.

```python
provider = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.example.com/v1",
    model="gpt-4.1",
)
```

## Google endpoint

Use `GoogleEndpointProvider` for Gemini models on Google's Generative Language
API or compatible Vertex AI endpoints. Choose it for Gemini vision-language
workloads, including image and video-style prompting.

```python
provider = mdr.inference.GoogleEndpointProvider(
    model="gemini-2.5-pro",
)
```

## Anthropic endpoint

Use `AnthropicEndpointProvider` for Anthropic-hosted Claude models. Choose it
when you want Claude's instruction following or long-context behavior while
keeping the rest of the pipeline code provider-agnostic.

```python
provider = mdr.inference.AnthropicEndpointProvider(
    model="claude-sonnet-4",
)
```

## vLLM

Use `VLLMProvider` with cloud launch when a pipeline should run against a
managed vLLM service for an open-weight model. Refiner provisions the vLLM
service as part of the cloud job and routes worker requests to it. You do not
need to assign GPUs to the pipeline workers for the model server; Refiner Cloud
manages service resources separately from worker resources.

```python
provider = mdr.inference.VLLMProvider(
    model="Qwen/Qwen3.5-4B",
)
```

As of June 3, 2026, managed vLLM supports the following models. To request
another model, contact us through [macrodata.co/contact](https://macrodata.co/contact).

| Model | Context length | Multimodality support |
|---|---:|---|
| `Qwen/Qwen3.5-9B` | 32,768 tokens | Image + text |
| `google/gemma-4-E4B-it` | 32,768 tokens | Image + text |
| `nvidia/Cosmos-Reason2-8B` | 32,768 tokens | Image + video + text, capped at 16 images or 1 video per prompt |
| `rednote-hilab/dots.mocr` | 32,768 tokens | Image + text |
| `Qwen/Qwen3.5-4B` | 32,768 tokens | Image + text |
| `robometer/Robometer-4B` | 4,096 tokens | Image + text, capped at 16 images per prompt |

## Provider options

Provider options are typed objects for model-specific request settings. They are
passed as `provider_options` to `generate_text(...)`.

Use provider options sparingly. Keep pipeline code portable unless you need a
specific model feature.

## Related pages

- [Generate Text](generate-text.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
