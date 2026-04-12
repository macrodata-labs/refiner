---
title: "Observability"
description: "Cloud observability available when running Refiner on Macrodata Cloud"
---

Refiner cloud jobs report runtime lifecycle and telemetry to Macrodata Cloud.

## Scope

This page applies to [`launch_cloud(...)`](launchers.md) only.

- `launch_local(...)` does not provide Macrodata Cloud observability
- cloud observability is available as part of Macrodata Cloud execution

## What Cloud Reports

Cloud execution reports:

- worker start / finish
- shard claim / finish
- stage finish
- job finish
- OTEL logs
- OTEL metrics

## User Metrics API

You can emit user metrics inside pipeline code:

```python
import refiner as mdr

def tag_row(row: mdr.Row) -> mdr.Row:
    row.log_throughput("rows_seen", 1, unit="rows")
    row.log_histogram("latency_ms", 42.5, unit="ms")
    return row.update(tagged=True)

pipeline = mdr.read_parquet("data/*.parquet").map(tag_row)
```

Available helpers:

- `row.log_throughput(...)`
- `row.log_histogram(...)`
- `mdr.log_throughput(...)`
- `mdr.log_gauge(...)`
- `mdr.log_gauges(...)`
- `mdr.log_histogram(...)`
- `mdr.register_gauge(...)`

When running on Macrodata Cloud, these metrics are exported into the cloud observability pipeline.

## Logging

Application logs emitted during cloud execution are forwarded into cloud observability.

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
result = pipeline.launch_cloud(name="train-data-build", num_workers=4)
```

## Notes

- cloud observability is an internal cloud-runtime integration, not part of the local OSS worker surface
- user metrics flush at runtime-managed boundaries in cloud workers

## Related Pages

- [Launchers](launchers.md)
- [CLI](cli.md)
- [Robotics](robotics.md)
