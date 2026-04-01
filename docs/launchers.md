---
title: "Launchers"
description: "Run Refiner pipelines as local or cloud jobs"
---

Launchers are the job-level execution entry points for Refiner.

## Local Launcher

Use `launch_local(...)` to run a pipeline as one or more worker subprocesses on the current machine.

```python
import refiner as mdr

pipeline = (
    mdr.read_parquet("data/*.parquet")
    .map(lambda row: {"x": row["x"]})
    .write_jsonl("out/")
)

stats = pipeline.launch_local(
    name="local-job",
    num_workers=2,
    cpus_per_worker=2,
    gpus_per_worker=1,
)
```

Returned stats include:

- `job_id`
- `workers`
- `claimed`
- `completed`
- `failed`
- `output_rows`

### Local runtime backends

`launch_local(...)` supports:

- `runtime_backend="auto"`: use platform reporting when auth is available, otherwise fall back to the filesystem runtime
- `runtime_backend="platform"`: require Macrodata platform lifecycle integration
- `runtime_backend="file"`: use only the local filesystem runtime

### Local notes

- workers always run as subprocesses, even when `num_workers=1`
- `cpus_per_worker` is optional CPU pinning when the OS supports affinity
- `gpus_per_worker` is optional local GPU partitioning via `CUDA_VISIBLE_DEVICES`
- local runtime files live under `<workdir>/runs/<job_id>/...`

## Cloud Launcher

Use `launch_cloud(...)` to submit the compiled pipeline to Macrodata Cloud.

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(lambda row: row["lang"] == "en")
    .write_parquet("hf://datasets/macrodata/my-output")
)

result = pipeline.launch_cloud(
    name="cloud-job",
    num_workers=8,
    cpus_per_worker=2,
    mem_mb_per_worker=4096,
    gpus_per_worker=1,
    gpu_type="h100",
)
```

Returned result includes:

- `job_id`
- `stage_index`
- `status`

### Cloud options

- `num_workers`: requested logical worker count
- `cpus_per_worker`: scheduler hint for worker CPU sizing
- `mem_mb_per_worker`: scheduler hint for worker memory sizing
- `gpus_per_worker`: scheduler hint for worker GPU count
- `gpu_type`: scheduler hint for worker GPU type
- `sync_local_dependencies`: whether to install the submitting environment's dependencies into the cloud image
- `secrets`: env vars sent as secrets
- `env`: env vars sent as plain runtime environment values

Requesting a specific `gpu_type` is a cloud-only feature. When requesting GPUs in
the cloud, you must set both `gpus_per_worker` and `gpu_type`.

`secrets` and `env` are both mounted into the cloud runtime, but only `secrets` participate in captured-code redaction.

## Authentication

Platform integration uses the same auth flow everywhere:

1. `MACRODATA_API_KEY`
2. local credential file from [`macrodata login`](cli.md)

Behavior:

- `launch_local(...)` can run without auth, but platform reporting becomes unavailable
- `launch_cloud(...)` requires auth

## Related Pages

- [CLI](cli.md)
- [Observability](observability.md)
- [Robotics](robotics.md)
