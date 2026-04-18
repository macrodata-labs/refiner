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
- `REFINER_LOCAL_LOGS` controls the live local console stream when set:
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

## Internal Notes

- the local launcher keeps worker result JSON on `stdout` reserved for launcher-to-worker control flow and suppresses that final payload from the live log pane

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

`secrets` and `env` are both mounted into the cloud runtime, but only `secrets` participate in captured-code redaction.

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
