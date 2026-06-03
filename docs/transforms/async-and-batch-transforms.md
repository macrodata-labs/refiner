---
title: "Async and Batch Transforms"
description: "Use map_async and batch_map for concurrent I/O and batched model work"
---

# Async And Batch Transforms

Use async and batch transforms when a row transform would be too slow one row at
a time.

## `map_async`

`map_async` runs an async function over rows with bounded concurrency.

```python
async def classify_episode(row):
    label = "long" if row.num_frames > 300 else "short"
    return row.update({"label": label})


pipeline = pipeline.map_async(
    classify_episode,
    max_in_flight=32,
    preserve_order=True,
)
```

Use `max_in_flight` to match provider rate limits, memory limits, or open file
limits.

## Inference Helpers

Most model calls should use Refiner's inference helpers instead of calling an
API directly:

```python
pipeline = pipeline.map_async(
    mdr.inference.generate_text(
        fn=annotate,
        provider=mdr.inference.OpenAIResponsesProvider(model="gpt-4.1"),
        max_concurrent_requests=64,
    ),
    max_in_flight=64,
)
```

See [Generate Text](../inference/generate-text.md).

## `batch_map`

`batch_map` receives a list of rows and returns rows.

```python
def add_batch_rank(rows):
    for index, row in enumerate(rows):
        yield row.update({"batch_rank": index})


pipeline = pipeline.batch_map(add_batch_rank, batch_size=32)
```

Use this for perception pipelines or model APIs that naturally process batches.

## Order And Backpressure

| Setting | Effect |
| --- | --- |
| `preserve_order=True` | Output order follows input order. |
| `preserve_order=False` | Faster rows may be emitted first. |
| Smaller `max_in_flight` | Lower memory and lower external pressure. |
| Larger `max_in_flight` | More concurrency, but easier to hit rate limits. |

## Related Pages

- [Subtask Annotation](../episode-operations/subtask-annotation.md)
- [Hand Tracking](../episode-operations/hand-tracking.md)
- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
