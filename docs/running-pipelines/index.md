---
title: "Running Pipelines"
description: "How Refiner pipelines execute locally and on Macrodata Cloud"
---

# Running Pipelines

A Refiner pipeline can be evaluated three ways:

| Mode | Use it when | Entry point |
| --- | --- | --- |
| In process | You are inspecting rows, testing a transform, or debugging quickly. | `iter_rows()`, `take()`, `materialize()` |
| Local launch | You want real worker/shard behavior on your machine. | `launch_local(...)` |
| Cloud launch | You want managed workers, logs, metrics, manifests, and scalable resources. | `launch_cloud(...)` |

Start with [In-Process Debugging](in-process-debugging.md), then move to
[Local Launcher](local-launcher.md) or [Cloud Launcher](cloud-launcher.md).

## Execution Terms

| Term | Meaning |
| --- | --- |
| Source | The reader that plans shards and emits input rows or tables. |
| Shard | A unit of source work assigned to a worker. |
| Worker | A process that claims shards, runs transforms, and writes output. |
| Stage | A contiguous execution segment. Writer reducers may add later stages. |
| Sink | A writer attached to the end of the pipeline. |

For input planning details, see [Reader Model](../reading-data/reader-model.md)
and [Sharding](../reading-data/sharding.md).

## Related Pages

- [In-Process Debugging](in-process-debugging.md)
- [Local Launcher](local-launcher.md)
- [Cloud Launcher](cloud-launcher.md)
- [Resources, GPUs, and Services](resources-gpus-and-services.md)
- [Observability](observability.md)

