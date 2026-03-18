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

## Sinks Are Ignored By Local Iteration

If a pipeline has a sink attached, `iter_rows()`, `materialize()`, and `take()` still behave as read/debug helpers and do not write output files.

Sink writes happen only in launcher/worker execution such as `launch_local(...)`.

## Vectorized Memory Guardrail

Use a max-bytes cap when running vectorized segments to reduce OOM risk:

```python
pipeline = pipeline.with_max_vectorized_block_bytes(64 * 1024 * 1024)
```

When set, Refiner shrinks row-to-Arrow chunk sizes during that run if an Arrow allocation fails, and retries with smaller chunks.

## Behavior Across Shards

Local iteration consumes the source stream continuously, so downstream batch steps can consume data that spans multiple shards.

## Async Debugging

If you need to debug blocking video/cache operations without any pipeline threadpool offloads, set `ASYNC_NO_THREADING=1` before running Refiner:

```bash
ASYNC_NO_THREADING=1 uv run examples/lerobot/motion_trim.py
```

This forces helpers that normally use `run_in_executor(...)` or `asyncio.to_thread(...)` to run inline on the async runtime thread instead. It is a debugging tool, not a performance mode.

## Internal Notes

- Local execution compiles pipeline steps into row/vector segments once per pipeline instance and reuses that plan across repeated runs.
- Row/UDF segments emit row blocks; when a vectorized segment follows, those rows are converted back to Arrow blocks at the segment boundary.
- Async row steps use the same execution engine and run through a shared process-local asyncio runtime.
- `ASYNC_NO_THREADING=1` disables pipeline threadpool offloads for blocking helpers, but it does not disable the shared asyncio runtime thread itself.
- `take(n)` stops early without forcing full-stream materialization.
