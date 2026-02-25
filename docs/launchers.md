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

## Observability (Env-Driven)

Local launches automatically attempt Macrodata Observer lifecycle reporting when a Macrodata API key is available.

- Key lookup order:
  - `MACRODATA_API_KEY`
  - local key file from `macrodata login`
- If no key is found, the launch still runs and prints a warning explaining how to enable observability.

This integration currently reports job/stage/worker/shard lifecycle events (no pulse metrics yet).

## Launch Result

`launch()` returns aggregate stats:

- `run_id`
- `workers`
- `claimed`
- `completed`
- `failed`
- `output_rows`

## Internal Notes

- Interface is intentionally minimal right now: launcher construction (`__init__(pipeline, name, ...)`) and `launch()`.
- Local launcher uses filesystem ledger coordination and subprocess worker execution under the hood.
- Worker subprocesses load pipeline payloads serialized with `cloudpickle`.
- CPU pinning is opt-in (`cpus_per_worker`) and does not enforce thread-count limits by default.
- Observability is auto-enabled from platform auth state (env/local key) without an explicit launcher flag.
- Slurm and Ray launchers are expected to implement the same minimal launcher shape later.
