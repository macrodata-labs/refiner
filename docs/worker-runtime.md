---
title: "Worker Runtime"
description: "How Refiner workers execute shards and talk to runtime lifecycle backends"
---

Use `Worker` directly when you need explicit shard-claiming execution instead of the higher-level launchers.

## Basic Usage

```python
import refiner as mdr

worker = mdr.Worker(
    rank=0,
    runtime_lifecycle=runtime_lifecycle,
    pipeline=pipeline,
)
stats = worker.run()
```

## What `run()` Does

`Worker.run()`:

1. claims shards from the runtime lifecycle backend
2. streams rows into the compiled pipeline
3. sends periodic heartbeats
4. marks shards complete on success
5. marks shards failed on exception
6. returns aggregate worker stats

## Runtime Lifecycle Backends

Worker lifecycle implementations live under `src/refiner/worker/lifecycle/`.

Current backends:

- local filesystem runtime
- Macrodata platform runtime

The launchers choose the backend for you in normal use.

## When To Use Worker Directly

Use `Worker` directly when you are:

- testing runtime lifecycle behavior
- embedding Refiner into another launcher/orchestrator
- debugging shard claim / finish behavior

For normal user-facing job execution, prefer [`launch_local(...)` or `launch_cloud(...)`](launchers.md).

## Notes

- worker subprocess bootstrapping lives in `src/refiner/worker/entrypoint.py`
- lifecycle and worker cleanup are intentionally centralized in the worker runtime
- worker resources and telemetry helpers live under `src/refiner/worker/resources/` and `src/refiner/worker/metrics/`

## Related Pages

- [Launchers](launchers.md)
- [Readers and sharding](readers-and-sharding.md)
- [Observability](observability.md)
