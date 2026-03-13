---
title: "Launchers"
description: "Run pipelines as jobs through launcher objects"
---

Refiner launchers provide a job-level execution entry point.

## Local Launcher (Current)

Run a pipeline locally as a job:

```python
import refiner as mdr

pipeline = (
    mdr.read_parquet("data/*.parquet")
    .map(lambda r: {"x": r["x"]})
)

stats = pipeline.launch_local(
    name="my-job",
    num_workers=2,
    cpus_per_worker=4,
)
```

`name` is required and identifies the launched job run.
`cpus_per_worker` is optional and pins each worker to a disjoint CPU set when supported by the OS.

## Platform Integration (Env-Driven)

Local launches automatically attempt Macrodata platform lifecycle reporting when a Macrodata API key is available.

- Key lookup order:
  - `MACRODATA_API_KEY`
  - local key file from `macrodata login`
- If no key is found, the launch still runs and prints a warning explaining how to enable platform integration.

This integration reports job/stage/worker/shard lifecycle events and user-emitted OTEL metrics (`mdr.log_counter`, `mdr.log_gauge`, `mdr.log_histogram`).

## Launch Result

`launch()` returns aggregate stats:

- `job_id`
- `workers`
- `claimed`
- `completed`
- `failed`
- `output_rows`

## Internal Notes

- Interface is intentionally minimal right now: launcher construction (`__init__(pipeline, name, ...)`) and `launch()`.
- `LocalLauncher` always launches worker subprocesses, even for `num_workers=1`. There is no in-process special case anymore.
- Worker subprocesses load pipeline payloads serialized with `cloudpickle`.
- Local runtime work files live under `<workdir>/runs/<job_id>/...`, where `workdir` comes from `REFINER_WORKDIR` or the cache default.
- Runtime lifecycle backends are:
  - `platform`: backend job/stage/worker/shard reporting through the Macrodata API
  - `file`: local filesystem shard coordination under the workdir
  - `auto`: try platform first, otherwise fall back to file
- Example failure scenario script: `examples/local_launcher_worker0_crash_demo.py` intentionally exits worker rank `0` after its first successful shard.
- CPU pinning is opt-in (`cpus_per_worker`) and does not enforce thread-count limits by default.
- Observability is auto-enabled from platform auth state (env/local key) without an explicit launcher flag.
- Slurm and Ray launchers are expected to implement the same minimal launcher shape later.
