---
title: "Jobs, Logs, and Metrics"
description: "Inspect cloud jobs from the Macrodata CLI"
---

# Jobs, Logs, And Metrics

## List Jobs

```bash
macrodata jobs list
```

## Inspect One Job

```bash
macrodata jobs get job_123
```

## Attach To A Running Job

```bash
macrodata jobs attach job_123
```

## Follow Logs

```bash
macrodata jobs logs job_123 --follow
```

## Stage Metrics

```bash
macrodata jobs metrics job_123 0
```

## Resource Metrics

```bash
macrodata jobs resource-metrics job_123 0
```

## Cancel

```bash
macrodata jobs cancel job_123
```

## Related Pages

- [Observability](../running-pipelines/observability.md)
- [Cloud Jobs and Files](../platform/cloud-jobs-and-files.md)

