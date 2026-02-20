---
title: "Worker Runtime"
description: "How worker execution interacts with ledger state and shard processing"
---

Use `Worker` when running shard-claiming execution with ledger state management.

## Basic Worker Usage

```python
import refiner as mdr

worker = mdr.Worker(rank=0, ledger=ledger, pipeline=pipeline)
stats = worker.run()
```

## What `run()` Does

1. Claims shards from the ledger.
2. Streams shard rows into fused pipeline execution.
3. Sends periodic heartbeats.
4. Marks shards complete on success, or failed on exception.
5. Returns summary stats (`claimed`, `completed`, `failed`, `output_rows`).

## When to Use It

- Use local pipeline iteration for notebook/dev workflows.
- Use `Worker` when you need explicit shard lifecycle coordination.

## Internal Notes

- Worker code lives in `src/refiner/runtime/worker.py`.
- A compatibility re-export still exists at `src/refiner/worker.py`.
