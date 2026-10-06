---
title: "Local launcher"
description: "Run Refiner pipelines with local worker processes"
---

# Local launcher

Use the local launcher to run workers and process shards on your own machine.

```python
pipeline.launch_local(
    name="debug-local",
    num_workers="auto",
)
```

With `num_workers="auto"`, Refiner starts one local worker for each shard in
the current stage. An empty stage starts no workers. Pass a positive
integer to cap execution at a fixed number of worker processes.

The local launcher is useful for:

- verifying a writer on a small dataset
- checking that a transform is safe across multiple shards
- checking resource requirements on your hardware

## Run ordered stages locally

Run stages in order:

```python
workflow = prepare.as_stage(
    name="prepare",
    num_workers=4,
).then(
    publish,
    name="publish",
    num_workers=1,
)

workflow.launch_local(name="staged-local")
```

Each stage is a complete pipeline with its own source and sink. The next stage
starts only after the previous stage completes successfully.

Local runs support per-stage `num_workers` and `gpu` settings. Local execution does not accept CPU and memory limits; remove `cpus_per_worker` and
`mem_mb_per_worker` before calling `launch_local(...)`.

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

Local GPU assignment controls `CUDA_VISIBLE_DEVICES` for worker processes.

## Environment variables

| Variable | Effect |
| --- | --- |
| `REFINER_WORKDIR` | Sets the local worker working directory. Use an absolute path. |
| `XDG_CACHE_HOME` | Sets the cache root used when `REFINER_WORKDIR` is unset. |
| `CUDA_VISIBLE_DEVICES` | Selects the GPU IDs available to local workers. Refiner sets this for each worker after assigning GPUs. |

Worker files default to `$XDG_CACHE_HOME/macrodata/refiner` or
`~/.cache/macrodata/refiner`. Set `REFINER_WORKDIR` to place them on a disk
with enough space for your dataset processing.

## Internal Notes

Spark and Beam/Dataflow schedule partitions through distributed executors;
Daft and Ray/Ray Data schedule partition tasks through their own runtimes; and
Hugging Face Datasets exposes explicit process counts for local transforms.
Refiner instead uses ledger-backed shard claims as its scheduling unit, so
`num_workers` sets fixed local process concurrency while each worker repeatedly
claims available shards. Shard count controls work granularity, not the number
of local worker processes.
