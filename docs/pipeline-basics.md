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
# or: mdr.read_lerobot("s3://bucket/lerobot-dataset")
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

For sync steps that need async I/O, offload a coroutine to the shared island runtime:

```python
import asyncio
import refiner as mdr


async def fetch_len(path: str) -> int:
    await asyncio.sleep(0.01)
    return len(path)


pipeline = pipeline.map(
    lambda r: {"path_len": mdr.submit(fetch_len(r["path"])).result()}
)
```

For file/video handles (for example from `read_lerobot`), use the built-in hydrator:

```python
pipeline = pipeline.flat_map(mdr.hydrate_file(columns="observation.images.main"))
```

This attaches a lazy `VideoFile` handle by default. Use `video_hydration="bytes"` only when you need full video bytes in memory.

Hydration already performs bounded in-flight buffering; tune it with `max_in_flight`:

```python
pipeline = pipeline.flat_map(
    mdr.hydrate_file(
        columns="observation.images.main",
        max_in_flight=8,
    )
)
```

For LeRobot episode-dict transforms, wrap functions with `convert_le_robot_fc(...)`:

```python
pipeline = pipeline.map(
    mdr.convert_le_robot_fc(lambda ep: {"episode_index": int(ep["episode_index"]) + 1})
)
```

## Step Output Contract

`map` can return:

- `Row`: replace current row
- `Mapping[str, Any]`: merged/normalized into a row

Use `.filter(...)` or `.flat_map(...)` for row dropping/expansion.

## Internal Notes

- Batch steps are triggered when enough rows are available for that step’s `batch_size`; tail rows are flushed at end-of-stream.
- `mdr.submit(...)` targets a shared per-worker async runtime and returns a `Future`.
