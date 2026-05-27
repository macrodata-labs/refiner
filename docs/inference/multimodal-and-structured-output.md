---
title: "Multimodal and Structured Output"
description: "Send media to models and validate structured model responses"
---

# Multimodal And Structured Output

Refiner can pass text, images, videos, and files to supported providers.

## Image Content

```python
async def describe_first_frame(row, generate_text):
    frame = await anext(row.videos["front"].iter_numpy_frames())
    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the robot state."},
                    {"type": "image", "image": frame},
                ],
            }
        ],
    )
    return row.update({"description": response.text})
```

## Video Content

Some providers accept video content. For providers that do not, use sampled
frames or contact sheets. The subtask annotation operation uses contact sheets;
see [Subtask Annotation](../episode-operations/subtask-annotation.md).

## Structured Output

Use a Pydantic model when you need validated fields:

```python
class Segment(BaseModel):
    start_sec: float
    end_sec: float
    label: str


class Segments(BaseModel):
    segments: list[Segment]
```

Then pass `schema=Segments` to `generate_text`.

## Raw Payloads

Use `raw_payload` only when a provider requires options that are not represented
by the common interface:

```python
response = await generate_text(
    raw_payload={
        "messages": [{"role": "user", "content": "Return JSON."}],
        "response_format": {"type": "json_object"},
    }
)
```

Prefer the typed interface unless you need provider-specific behavior.

