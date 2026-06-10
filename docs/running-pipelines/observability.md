---
title: "Observability"
description: "Logs, user metrics, resource metrics, and job progress for Refiner runs"
---

# Observability

Cloud jobs report progress, logs, and metrics while workers run. Use these
signals to answer three questions:

1. Is the job making progress?
2. Which stage or worker is slow or failing?
3. Are custom pipeline metrics behaving as expected?

## User metrics

Transforms can emit counters, gauges, and histograms:

```python
def count_frames(row):
    row.log_throughput("frames_seen", row.num_frames, unit="frames")
    row.log_histogram("frames_per_episode", row.num_frames, unit="frames")
    return row
```

For metrics outside a row method, use the top-level API:

```python
import refiner as mdr

mdr.log_gauge("queue_depth", 12, unit="items")
```

## Logs

Use the worker logger for structured job logs:

```python
import refiner as mdr

def transform(row):
    mdr.logger.info("processing episode", episode_id=row.episode_id)
    return row
```

Worker failure logs include the operation that failed when a shard is being
finalized. For example, finalization errors distinguish sink completion, user
metrics flush, and shard lifecycle completion. Platform request timeouts include
the HTTP method, API path, and timeout duration so you can tell which API call
did not return. Stage, worker, and shard lifecycle reports retry transient
platform errors up to three times with exponential backoff before the worker
treats the report as failed.

## CLI inspection

```bash
macrodata jobs list
macrodata jobs get job_123
macrodata jobs logs job_123 --follow
macrodata jobs metrics job_123 0
macrodata jobs resource-metrics job_123 0
```

See [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md).

## Related pages

- [Cloud Launcher](cloud-launcher.md)
- [Platform Jobs and Files](../platform/cloud-jobs-and-files.md)
- [Resources, GPUs, and Services](resources-gpus-and-services.md)
