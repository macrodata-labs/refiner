---
title: "Pipeline Basics"
description: "How to construct and compose Refiner pipelines"
---

A `RefinerPipeline` is built from a source reader plus ordered processing steps.

## Create a Pipeline

Use one of the reader entry points:

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet")
```

## Add Processing Steps

Use `.map(...)` for row transforms and `.batch_map(...)` for batch transforms.

```python
pipeline = (
    pipeline
    .map(lambda r: {"x": r["x"] + 1})
    .batch_map(lambda batch: [row for row in batch if row["x"] > 0], batch_size=64)
)
```

Use `.flat_map(...)` for row-level expansion (`0..N` output rows per input row):

```python
pipeline = pipeline.flat_map(lambda r: [r, {"x": r["x"] * 10}])
```

Use `.filter(...)` for row predicates:

```python
pipeline = pipeline.filter(lambda r: r["x"] > 0)
```

## Step Output Contract

`map` can return:

- `Row`: replace current row
- `Mapping[str, Any]`: merged/normalized into a row

Use `.filter(...)` or `.flat_map(...)` for row dropping/expansion.

## Internal Notes

- Batch steps are triggered when enough rows are available for that step’s `batch_size`; tail rows are flushed at end-of-stream.
- Execution currently uses per-step queues in `RefinerPipeline.execute_rows(...)`.
