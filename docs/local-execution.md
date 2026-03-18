---
title: "Local Execution"
description: "Iterate over Refiner pipelines in-process"
---

Refiner supports an in-process execution mode for quick iteration and debugging.

## Iterate Lazily

```python
for row in pipeline:
    print(row)
```

This executes the pipeline incrementally as rows are consumed.

## Materialize Everything

```python
rows = pipeline.materialize()
```

Use this when you explicitly want the full result in memory.

## Inspect A Sample

```python
sample = pipeline.take(10)
```

`take(n)` stops early and does not force a full run.

## Sinks In Local Iteration

If a pipeline has a sink attached:

- `iter_rows()`
- `materialize()`
- `take()`

still behave as read/debug helpers and do not write sink output.

Sink writes happen in launched worker execution such as `launch_local(...)` or `launch_cloud(...)`.

## Vectorized Memory Guardrail

You can limit Arrow block sizing during vectorized execution:

```python
pipeline = pipeline.with_max_vectorized_block_bytes(64 * 1024 * 1024)
```

This is a vectorized chunk-size guardrail, not a job-level memory limit.

## Notes

- local iteration runs in-process, not through worker subprocesses
- downstream steps can observe data spanning multiple source shards
- async steps still use the shared local asyncio runtime

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Expression transforms](expression-transforms.md)
- [Launchers](launchers.md)
