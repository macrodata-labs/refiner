---
title: "In-Process Debugging"
description: "Iterate over Refiner pipelines directly in-process"
---

Use in-process execution when you want to debug pipeline logic without going through the launcher/runtime path.

## Iterate Lazily

```python
for row in pipeline:
    print(row)
```

## Materialize Everything

```python
rows = pipeline.materialize()
```

## Inspect A Sample

```python
sample = pipeline.take(10)
```

## Sinks In In-Process Mode

If a pipeline has a sink attached, these helpers:

- `iter_rows()`
- `materialize()`
- `take()`

still behave as debugging helpers and do not write sink output.

Sink writes happen in launched execution such as `launch_local(...)` or `launch_cloud(...)`.

## Vectorized Memory Guardrail

```python
pipeline = pipeline.with_max_vectorized_block_bytes(64 * 1024 * 1024)
```

This is a chunk-size guardrail for vectorized execution, not a job-level memory limit.

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Transforms](transforms.md)
- [Expressions](expressions.md)
- [Launchers](launchers.md)
