---
title: "Observability"
description: "What Refiner reports to Macrodata during launched execution"
---

When Macrodata auth is available, Refiner reports runtime lifecycle and telemetry back to the platform.

## How It Enables

Auth lookup order:

1. `MACRODATA_API_KEY`
2. local key file from [`macrodata login`](cli.md)

If auth is unavailable:

- `launch_local(...)` still runs, but without platform reporting
- `launch_cloud(...)` cannot submit

## What Gets Reported

Current launched execution reports:

- job creation
- manifest capture
- stage shard registration
- worker start / finish
- shard claim / heartbeat / finish
- stage finish
- job finish
- OTEL logs
- OTEL metrics

## User Metrics API

You can emit user metrics inside pipeline code:

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet").map(
    lambda row: (
        mdr.log_throughput("rows_seen", 1, shard_id=str(row["__shard_id"]), unit="rows"),
        mdr.log_gauge("batch_size", 128, unit="rows"),
        mdr.log_histogram("latency_ms", 42.5, shard_id=str(row["__shard_id"]), unit="ms"),
        row,
    )[-1]
)
```

Available helpers:

- `mdr.log_throughput(...)`
- `mdr.log_gauge(...)`
- `mdr.log_gauges(...)`
- `mdr.log_histogram(...)`

If telemetry is unavailable, these calls are no-ops.

## Logging

Loguru records are forwarded into platform OTLP logs when observability is enabled.

```python
from loguru import logger

logger.info("started processing")
```

## Example Flow

```bash
macrodata login
```

```python
import refiner as mdr

pipeline = mdr.read_parquet("data/*.parquet").write_jsonl("out/")
stats = pipeline.launch_local(name="train-data-build", num_workers=4)
```

## Notes

- current Refiner launches typically submit a single compiled plan, but lifecycle still reports jobs, stages, workers, and shards explicitly
- user metrics flush on shard end
- worker resource metrics cover CPU, memory, and network observers

## Related Pages

- [Launchers](launchers.md)
- [CLI](cli.md)
- [Robotics](robotics.md)
