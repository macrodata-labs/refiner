---
title: "Resources, GPUs, and services"
description: "CPU, memory, GPU, and runtime service configuration for launched Refiner jobs"
---

# Resources, GPUs, and services

> [!WARNING]
>
> **Early access:** Cloud GPUs and managed runtime services are currently
> available to approved Macrodata Cloud partners. Local Refiner resource
> settings remain available to all open-source users.

Resource settings belong on launch calls, not inside transform functions. This
keeps the pipeline logic portable between local debugging and cloud execution.

## CPU and memory

```python
pipeline.launch_cloud(
    name="cpu-heavy-transform",
    num_workers=16,
    cpus_per_worker=8,
    mem_mb_per_worker=32768,
)
```

Use more memory for video-heavy readers and writers, large frame tables, and
large vectorized batches.

## GPUs

```python
pipeline.launch_cloud(
    name="gpu-inference",
    num_workers=4,
    gpu=mdr.GPU(type="h100", count=1),
)
```

GPU requests apply per worker. If your transform uses a model directly inside
the worker, make sure the worker count and GPU count match the model loading
pattern you want.

Macrodata Cloud supports these GPU types:

- `h100`
- `l40s`
- `a100`
- `a10`
- `l4`
- `t4`

## Runtime services

Some inference workloads are better served by a runtime service, such as vLLM,
instead of loading a model inside every transform worker.

```python
provider = mdr.inference.VLLMProvider(
    model="Qwen/Qwen3.5-9B",
    config="throughput",
)
```

When a provider requires a service, Refiner can include that service in the
cloud runtime plan. See [Providers and vLLM](../inference/inference_providers.md).

## Choosing worker count

| Workload | Worker count guidance |
| --- | --- |
| Reading many small files | More workers usually helps. |
| Writing videos | Start modestly; video encoding can saturate CPU and I/O. |
| VLM/API calls | Match worker count to provider rate limits and `max_in_flight`. |
| Local GPU inference | Avoid more workers than available GPUs unless the model can share. |

## Related pages

- [Async and Batch Transforms](../transforms/async-and-batch-transforms.md)
- [Media Assets and Reducers](../writing-data/media-assets-and-reducers.md)
- [Observability](observability.md)
