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

### Local notes

- workers always run as subprocesses, even when `num_workers=1`
- local launch does not pin worker CPUs; if `num_workers` exceeds available CPUs, Refiner logs a warning and still launches the requested worker count
- `gpus_per_worker` optionally exposes a fixed number of visible GPU devices to each local worker
- `REFINER_LOGS` controls the live local console stream when set:
  - `all`: stream every worker log line
  - `none`: suppress live worker log output
  - `one`: stream one representative worker
  - `errors`: stream only `ERROR` and `CRITICAL` lines
- if `rundir` is reused, local launch skips shards already completed there
- when a local run fails or is interrupted, Refiner prints the `rundir` to reuse if you want to resume completed shards
- local run files live under `<workdir>/runs/<job_id>/...`
- worker Loguru output is written to `stage-<index>/logs/worker-<worker_id>.log` under the local rundir
- during local execution, the launcher tails per-worker log files
- on interactive terminals, Refiner redraws a pinned terminal header with job metadata, live runtime, and the latest worker log lines beneath it
- on non-interactive output, Refiner prints plain prefixed log lines

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
- `gpus_per_worker`: scheduler hint for GPU count per worker
- `gpu_type`: required when `gpus_per_worker` is set
- `sync_local_dependencies`: whether to install the submitting environment's dependencies into the cloud image
- `secrets`: env vars sent as secrets
- `env`: env vars sent as plain runtime environment values
- `resume_from_job_id`: explicitly resume from one exact prior cloud job
- `resume="latest-compatible"`: explicitly ask the control plane to pick the latest compatible prior job
- `resume_name`: optional run-name filter used with `resume="latest-compatible"`
- `resume_limit_to_me`: restrict latest-compatible lookup to jobs started by the authenticated user

`secrets` and `env` are both mounted into the cloud runtime, but only `secrets` participate in captured-code redaction.

### Resuming cloud work

Fresh cloud launch remains the default. Resume only triggers when you pass an explicit selector.

Use an exact prior job id when you already know which failed attempt you want to continue:

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .map(lambda row: {"x": row["x"]})
    .write_parquet("hf://datasets/macrodata/my-output")
)

result = pipeline.launch_cloud(
    name="cloud-job",
    num_workers=16,
    cpus_per_worker=4,
    resume_from_job_id="job_123",
)
```

Use `resume="latest-compatible"` when you want an explicit selector without hard-coding a job id:

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .map(lambda row: {"x": row["x"]})
    .write_parquet("hf://datasets/macrodata/my-output")
)

result = pipeline.launch_cloud(
    name="cloud-job",
    resume="latest-compatible",
    resume_name="cloud-job",
    resume_limit_to_me=True,
)
```

Resume behavior notes:

- resume never triggers from the output path or job name alone; you must opt in with `resume_from_job_id` or `resume="latest-compatible"`
- resumed cloud launches create a new attempt instead of mutating the old job
- if you omit `num_workers`, `cpus_per_worker`, `mem_mb_per_worker`, `gpus_per_worker`, and `gpu_type` on resume, the control plane can inherit the selected run's existing sizing
- if you pass any of those fields on resume, they are treated as explicit overrides for the new attempt

### Launched writer notes

- some sinks add a 1-worker reducer stage after the main writer stage to finalize or clean up shard outputs
- for launched file sinks such as `write_jsonl(...)` and `write_parquet(...)`, the output prefix should be dedicated to Refiner-managed files so the reducer stage can safely remove stale shard/worker outputs

## Authentication

See [Auth](auth.md) for credential lookup order. In practice:

- `launch_local(...)` can run without auth
- `launch_cloud(...)` requires auth

## Related Pages

- [Auth](auth.md)
- [CLI](cli.md)
- [Observability](observability.md)
- [Robotics](robotics.md)

## Internal Notes

- the local launcher keeps worker result JSON on `stdout` reserved for launcher-to-worker control flow and suppresses that final payload from the live log pane
