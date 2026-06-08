---
title: "Cloud Jobs and files"
description: "Track cloud jobs and understand file staging from the user perspective"
---

# Cloud Jobs and files

A cloud job is a submitted Refiner run owned by a workspace. Jobs have stages,
workers, logs, metrics, manifests, resource settings, optional services, and
output references.

Open jobs from [Jobs](/jobs).

## Job lifecycle

| State | Meaning |
| --- | --- |
| Queued | The job or stage is waiting to run. |
| Running | Workers are claiming shards and processing data. |
| Succeeded | All required stages finished. |
| Failed | At least one required stage failed. |
| Canceled | A user canceled the job. |

The jobs list shows status, progress, duration, starter, and an action to open
the job. The job detail page shows the graph, selected-stage detail, worker
observability, logs, metrics, manifest, and job metadata.

## Stages, workers, and shards

Refiner breaks a cloud run into stages. Each stage can have workers. Workers
claim shards, process rows or episodes, emit logs and metrics, and report
progress back to the platform.

Use the job page when you need to know:

- which stage is slow or failed
- which worker owns a failed shard
- whether rows, episodes, or files are still being processed
- whether resource usage suggests CPU, memory, network, or GPU pressure

## Files

Cloud jobs may upload files needed for execution and write files as output. From
a user perspective, this matters because:

- outputs should be written to durable paths you control, such as S3, GCS, or
  Hugging Face repositories
- private input or output locations require credentials through
  [Secrets and environment](secrets-and-environment.md)
- manifests can show which code and file references were submitted with a job
- the [Viewer](viewer.md) can inspect produced Parquet, JSON, and CSV outputs
  from workspace-accessible storage

## Inspect Jobs in the CLI

```bash
macrodata jobs list
macrodata jobs get job_123
macrodata jobs logs job_123 --follow
macrodata jobs metrics job_123 0
macrodata jobs resource-metrics job_123 0
```

Use `--json` for automation:

```bash
macrodata jobs get job_123 --json
```

See [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md).

## Links to other platform surfaces

| Surface | URL |
| --- | --- |
| Jobs list | [Jobs](/jobs) |
| Services | [Services](/services) |
| Billing | [Settings > Billing](/settings/billing) |
| Viewer | [Viewer](/viewer) |
| Secrets | [Settings > Secrets](/settings/secrets) |

## Related pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Observability](../running-pipelines/observability.md)
- [Manifests](manifests.md)
- [Viewer](viewer.md)
