---
title: "Observability"
description: "Send local launcher lifecycle events to Macrodata Observer"
---

Refiner local launches can report job/stage/worker/shard lifecycle events and user metrics to Macrodata Observer.

## How It Enables

Observability is enabled automatically when a Macrodata API key is available:

1. `MACRODATA_API_KEY` environment variable
2. Local key file created by `macrodata login`

If no key is available, the launch still runs and prints a warning with a login/env hint.

## What Is Reported (Current)

- Job creation (`/api/jobs/submit`)
- Stage shard registration
- Worker start / finish
- Shard start / finish
- Stage finish
- Job finish
- OTEL metrics emitted from worker runtime
- OTEL logs exported from Loguru

## User Metrics API

Use top-level helpers inside pipeline functions:

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet").map(
    lambda r: (
        mdr.log_counter("rows_seen", 1, str(r["__shard_id"]), unit="rows"),
        mdr.log_gauge("batch_size", 128, shard_id=str(r["__shard_id"]), unit="rows"),
        mdr.log_histogram("latency_ms", 42.5, shard_id=str(r["__shard_id"]), unit="ms"),
        r,
    )[-1]
)
```

Available helpers:

- `mdr.log_counter(label, value, shard_id, *, unit=None)` -> OTEL counter (`refiner.user.counter`)
- `mdr.log_gauge(label, value, shard_id, *, unit=None)` -> OTEL gauge (`refiner.user.gauge`)
- `mdr.log_histogram(label, value, shard_id, *, unit=None)` -> OTEL histogram (`refiner.user.histogram`)

For `log_histogram`, `unit` is emitted as OTEL point attribute `unit`.

`log_counter` requires `shard_id` for idempotency keying and shard-end flush behavior.
Backend dedupe key for counters should be `(job.id, stage.index, label, shard_id)`.
`__shard_id` is attached to rows automatically by the source runtime.
`step.index` is attached from runtime execution context:
- source/reader step uses index `0`
- pipeline transform steps use index `1..N`

If telemetry is unavailable (for example local iteration via `pipeline.iter_rows()` or no Observer credentials), these calls are no-op.


## Logging API

Refiner forwards Loguru records to Observer OTLP logs when observability is enabled.

```python
from loguru import logger

logger.info("started processing")
logger.info("batch finished")
```

No extra logger wiring is required in user code.

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
- User OTEL metrics export every 10 seconds and force flush on shard end.
- Worker resource OTEL metrics export every 10 seconds and force flush on worker end.
- Observer failures are fail-open in local launcher mode (processing continues; warnings are printed).
