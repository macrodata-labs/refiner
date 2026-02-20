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

## Behavior Across Shards

Local iteration consumes the source stream continuously, so downstream batch steps can consume data that spans multiple shards.

## Internal Notes

- Local execution paths use `iter_rows() -> execute_rows(source.read())`.
- `take(n)` stops early without forcing full-stream materialization.
