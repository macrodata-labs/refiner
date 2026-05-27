---
title: "Cloud Jobs and Files"
description: "Track Macrodata cloud jobs and understand cloud file usage"
---

# Cloud Jobs And Files

A cloud job is a submitted Refiner run. Jobs have stages, workers, logs, metrics,
and output files.

## Job Lifecycle

| State | Meaning |
| --- | --- |
| Queued | The job or stage is waiting to run. |
| Running | Workers are claiming shards and processing data. |
| Succeeded | All required stages finished. |
| Failed | At least one required stage failed. |
| Canceled | A user canceled the job. |

## Files

Cloud jobs may upload files for execution and write files as output. From a user
perspective, this matters because:

- outputs should be written to durable paths you control
- private input/output locations require credentials
- manifests can show which files were submitted with a job

## Inspect Jobs

```bash
macrodata jobs list
macrodata jobs get job_123
macrodata jobs logs job_123 --follow
```

See [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md).

