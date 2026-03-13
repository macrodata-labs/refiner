---
title: "Pipeline Basics"
description: "How to construct and compose Refiner pipelines"
---

A `RefinerPipeline` is built from a source plus ordered processing steps, with an optional sink.

## Create a Pipeline

Use one of the reader entry points:

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet")
```

Or wrap a custom source:

```python
pipeline = mdr.from_source(my_source)
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

Use `.map_async(...)` for async row transforms without changing the rest of the pipeline model:

```python
pipeline = pipeline.map_async(fetch_enrichment, max_in_flight=32)
```

## Add a Sink

Sinks mirror sources and live on the pipeline boundary. Today Refiner supports:

- `.write_jsonl(...)`
- `.write_parquet(...)`

```python
pipeline = pipeline.write_parquet("out/")
```

Attaching a sink changes launched worker execution only. Local row iteration helpers still return rows and do not write files.

## Step Output Contract

`map` can return:

- `Row`: replace current row
- `Mapping[str, Any]`: merged/normalized into a row

Use `.filter(...)` or `.flat_map(...)` for row dropping/expansion.

## Internal Notes

- Batch steps are triggered when enough rows are available for that step’s `batch_size`; tail rows are flushed at end-of-stream.
- Async row steps run inside a process-local asyncio runtime owned by the execution layer.
- Shard identity is propagated through execution so sinks and worker lifecycle completion stay aligned.
