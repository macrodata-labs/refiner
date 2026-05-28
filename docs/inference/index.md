---
title: "Inference"
description: "Use language, vision-language, structured, VLLM, and pooling inference in Refiner"
---

# Inference

Refiner inference helpers wrap provider calls so model work can run inside
pipelines with bounded concurrency, metrics, retries, and media handling.

| Page | Use it for |
| --- | --- |
| [Generate Text](generate-text.md) | Text and chat-style inference inside transforms. |
| [Multimodal and Structured Output](multimodal-and-structured-output.md) | Images, videos, files, and validated JSON output. |
| [Providers and VLLM](providers-and-vllm.md) | OpenAI-compatible, Google, Anthropic, and VLLM providers. |
| [Pooling](pooling.md) | Pooling/token-classification style inference used by reward scoring. |

For episode-level VLM workflows, see
[Subtask Annotation](../episode-operations/subtask-annotation.md) and
[Reward Scoring](../episode-operations/reward-scoring.md).

