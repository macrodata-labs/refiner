---
title: "Local Execution"
description: "Run pipelines lazily in-process for development and notebooks"
---

Refiner supports a local mode that runs pipeline execution in-process and yields rows lazily.

## Lazy Iteration

You can iterate directly over a pipeline:

```python
for row in pipeline:
    print(row)
```

This processes data incrementally and only computes as rows are requested.

## Materialize All Rows

Use `.materialize()` when you need the full result in memory:

```python
rows = pipeline.materialize()
```

## Partial Reads

Use `.take(n)` for quick inspection:

```python
sample = pipeline.take(10)
```

## Vectorized Memory Guardrail

Use a max-bytes cap when running vectorized segments to reduce OOM risk:

```python
pipeline = pipeline.with_max_vectorized_block_bytes(64 * 1024 * 1024)
```

When set, Refiner shrinks row-to-Arrow chunk sizes during that run if an Arrow allocation fails, and retries with smaller chunks.

## Behavior Across Shards

Local iteration consumes the source stream continuously, so downstream batch steps can consume data that spans multiple shards.

## Internal Notes

- Local execution compiles pipeline steps into row/vector segments once per pipeline instance and reuses that plan across repeated runs.
- Row/UDF segments emit row blocks; when a vectorized segment follows, those rows are converted back to Arrow blocks at the segment boundary.
- `take(n)` stops early without forcing full-stream materialization.
