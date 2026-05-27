---
title: "Manifests"
description: "Understand Refiner cloud job manifests"
---

# Manifests

A manifest records what Refiner submitted for a cloud job. It helps you answer:

- what code was submitted
- what dependencies were captured
- which pipeline plan ran
- which secrets and environment names were referenced

It should not contain secret values.

## Inspect A Manifest

```bash
macrodata jobs manifest job_123 --deps --code
```

Use manifests for reproducibility and debugging, not as a public artifact.

## What To Look For

| Field | Why it matters |
| --- | --- |
| Pipeline plan | Confirms the reader, stages, transforms, and writer. |
| Dependencies | Confirms model/runtime libraries available to workers. |
| Code | Confirms the submitted script or module. |
| Refiner version | Confirms which package version workers used. |

## Related Pages

- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)

