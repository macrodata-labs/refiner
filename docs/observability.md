---
title: "Observability"
description: "Send local launcher lifecycle events to Macrodata Observer"
---

Refiner local launches can report job/stage/worker/shard lifecycle events to Macrodata Observer.

## How It Enables

Observability is enabled automatically when a Macrodata API key is available:

1. `MACRODATA_API_KEY` environment variable
2. Local key file created by `macrodata login`

If no key is available, the launch still runs and prints a warning with a login/env hint.

## What Is Reported (Current)

- Job creation (`/api/jobs`)
- Stage shard registration
- Worker start / finish
- Shard start / finish
- Stage finish
- Job finish

Metrics / pulse reporting is not sent yet.

## User Flow

```bash
macrodata login
```

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet").map(lambda r: {"x": r["x"]})
stats = pipeline.launch_local(name="train-data-build", num_workers=4)
```

If a key is present, the run appears in Macrodata Observer with lifecycle progress.

## Internal Notes

- Current Refiner pipelines are submitted as a single stage (`stage_0`) in the Observer job plan.
- Refiner registers shard descriptors (`shard_id`, `path`, `start`, `end`) and uses `shard_id` for shard lifecycle events.
- Observer failures are fail-open in local launcher mode (processing continues; warnings are printed).

