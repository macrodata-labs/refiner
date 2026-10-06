---
title: "Inference providers"
description: "Configure model providers for Refiner inference"
---

# Inference providers

Providers define where inference requests are sent and how Refiner formats each
request for that backend. Use them with `generate_text(...)` and other inference
helpers when you want the same pipeline code to run against third-party APIs or model servers you operate.

Refiner supports OpenAI, OpenAI-compatible endpoints, Google Gemini, Anthropic
Claude, and OpenAI-compatible model servers.

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

## Self-hosted vLLM

Connect to a vLLM server you operate using `OpenAIEndpointProvider` and its
OpenAI-compatible endpoint. Configure the model server and its resources on
your own infrastructure.

## Provider options

Provider options are typed objects for model-specific request settings. They are
passed as `provider_options` to `generate_text(...)`.

Use provider options sparingly. Keep pipeline code portable unless you need a
specific model feature.

## Related pages

- [Generate Text](generate-text.md)
