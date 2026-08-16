---
title: "Running pipelines"
description: "How Refiner pipelines execute locally and on the Macrodata Cloud"
---

# Running pipelines

A Refiner pipeline can be evaluated three ways:

| Mode | Use it when | Entry point |
| --- | --- | --- |
| In process | You are inspecting rows, testing a transform, or debugging quickly. | `iter_rows()`, `take()`, `materialize()` |
| Local launch | You want real worker/shard behavior on your machine. | `launch_local(...)` |
| Cloud launch (early access) | You are an approved partner and want managed workers, logs, metrics, manifests, and scalable resources. | `launch_cloud(...)` |

Start with [In-Process Debugging](in-process-debugging.md), then move to
[Local Launcher](local-launcher.md) or [Cloud Launcher](cloud-launcher.md).
Cloud launch is currently available to approved partners; Refiner's in-process
and local modes remain open to all users.

## Execution terms

| Term | Meaning |
| --- | --- |
| Source | The reader that plans shards and emits input rows or tables. |
| Shard | A unit of source work assigned to a worker. |
| Worker | A process that claims shards, runs transforms, and writes output. |
| Stage | A contiguous execution segment. Writer reducers may add later stages. |
| Sink | A writer attached to the end of the pipeline. |

For input planning details, see [Reader Model](../reading-data/reader-model.md)
and [Sharding](../reading-data/sharding.md).

## Task shard claiming

Each worker processes `mdr.task(...)` shards sequentially and claims one shard
at a time by default. The worker does not claim another shard until the current
shard's transforms and sink output finish. To preserve unbounded shard claiming:

```python
import refiner as mdr

pipeline = mdr.task(
    run_partition,
    num_tasks=32,
    claim_shards_sequentially=False,
)
```

Reader pipelines remain unbounded by default, preserving their existing
cross-shard batching behavior.

## Related pages

- [In-Process Debugging](in-process-debugging.md)
- [Local Launcher](local-launcher.md)
- [Cloud Launcher](cloud-launcher.md)
- [Resources, GPUs, and Services](resources-gpus-and-services.md)
- [Observability](observability.md)
