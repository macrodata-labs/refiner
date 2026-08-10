---
title: "Local launcher"
description: "Run Refiner pipelines with local worker processes"
---

# Local launcher

Use the local launcher when you want worker and shard behavior without submitting
to the Macrodata Cloud.

```python
pipeline.launch_local(
    name="debug-local",
    num_workers=2,
)
```

The local launcher is useful for:

- verifying a writer on a small dataset
- checking that a transform is safe across multiple shards
- debugging resource assumptions before a cloud launch

## Run directory

Local runs write run metadata under a local run directory. Pass `rundir` when
you want a stable location:

```python
pipeline.launch_local(
    name="debug-local",
    num_workers=2,
    rundir=".runs/debug-local",
)
```

Reusing the same run directory allows local execution to resume completed shard
work where possible.

## Local GPUs

You can request GPUs for local workers:

```python
pipeline.launch_local(
    name="gpu-check",
    num_workers=1,
    gpu=mdr.GPU(type="h100", count=1),
)
```

Local GPU assignment controls `CUDA_VISIBLE_DEVICES` for worker processes. Cloud
GPU scheduling has additional options; see
[Resources, GPUs, and Services](resources-gpus-and-services.md).

## When to use Cloud instead

Use [Cloud Launcher](cloud-launcher.md) when you need:

- more workers than fit on your machine
- fast network access to remote datasets, buckets, and model artifacts
- managed logs and metrics
- workspace secrets
- resumable cloud jobs
- hosted runtime services such as vLLM
