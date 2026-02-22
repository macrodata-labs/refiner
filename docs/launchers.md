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
- Slurm and Ray launchers are expected to implement the same minimal launcher shape later.
