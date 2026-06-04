---
title: "Generate Text"
description: "Use generate_text inside Refiner async transforms"
---

# Generate Text

`generate_text` turns a function that receives a row and a model-call helper
into an async transform.

```python
from pydantic import BaseModel


class EpisodeLabel(BaseModel):
    label: str
    confidence: float


async def label_episode(row, generate_text):
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": f"Label this task: {row.task}",
            }
        ],
        schema=EpisodeLabel,
    )
    return row.update(response.object.model_dump())


pipeline = pipeline.map_async(
    mdr.inference.generate_text(
        fn=label_episode,
        provider=mdr.inference.OpenAIResponsesProvider(model="gpt-4.1"),
        max_concurrent_requests=64,
    ),
    max_in_flight=64,
)
```

## Function Shape

Your function receives:

| Argument | Meaning |
| --- | --- |
| `row` | The current Refiner row. |
| `generate_text` | Async helper used to call the configured provider. |

The function returns an updated row.

## Generation Parameters

```python
await generate_text(
    messages=[{"role": "user", "content": "Summarize the episode."}],
    temperature=0.1,
    max_tokens=200,
)
```

Default parameters can be attached when creating the transform:

```python
mdr.inference.generate_text(
    fn=label_episode,
    provider=provider,
    default_generation_params={"temperature": 0.1},
)
```

## Related Pages

- [Multimodal and Structured Output](multimodal-and-structured-output.md)
- [Providers and vLLM](inference_providers.md)
- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
