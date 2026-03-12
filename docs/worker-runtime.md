---
title: "Worker Runtime"
description: "How worker execution interacts with runtime lifecycle state and shard processing"
---

Use `Worker` when running shard-claiming execution with runtime lifecycle state management.

## Basic Worker Usage

```python
import refiner as mdr

worker = mdr.Worker(rank=0, runtime_lifecycle=runtime_lifecycle, pipeline=pipeline)
stats = worker.run()
```

## What `run()` Does

1. Claims shards from the runtime lifecycle backend.
2. Streams shard rows into fused pipeline execution.
3. Sends periodic heartbeats.
4. Marks shards complete on success, or failed on exception.
5. Returns summary stats (`claimed`, `completed`, `failed`, `output_rows`).

## When to Use It

- Use local pipeline iteration for notebook/dev workflows.
- Use `Worker` when you need explicit shard lifecycle coordination against platform or local file runtime.

## Internal Notes

- Worker runtime code lives in `src/refiner/worker/runner.py`.
- Runtime lifecycle implementations live in `src/refiner/worker/lifecycle/`.
- Worker resource helpers (CPU, memory, network metrics and pinning helpers) live in `src/refiner/worker/resources/`.
- Worker subprocess bootstrapping lives in `src/refiner/worker/entrypoint.py`.
