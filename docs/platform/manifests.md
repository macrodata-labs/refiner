---
title: "Manifests"
description: "Understand and inspect Refiner cloud job manifests"
---

# Manifests

A manifest records what Refiner submitted for a cloud job. It is the
reproducibility record for a run: code, dependency context, pipeline plan,
resource settings, and secret references.

Open a job from [Jobs](/jobs), then click the **Manifest** tab.

## What A Manifest Answers

Use a manifest to answer:

- what code was submitted
- what Python/runtime dependencies were captured
- which pipeline plan ran
- which reader, transforms, writer, stages, and resource settings were used
- which secret names and environments were referenced
- which Refiner version and runtime context workers saw

Manifest records should not contain secret values.

## Inspect From The CLI

```bash
macrodata jobs manifest job_123
macrodata jobs manifest job_123 --deps
macrodata jobs manifest job_123 --code
macrodata jobs manifest job_123 --json
```

Use `--deps` to show dependencies and `--code` to show captured script text.
Use `--json` for an agent, notebook, or CI job.

## What To Look For

| Field | Why it matters |
| --- | --- |
| Pipeline plan | Confirms the reader, stages, transforms, and writer. |
| Dependencies | Confirms model/runtime libraries available to workers. |
| Code | Confirms the submitted script or module. |
| Secret references | Confirms which workspace secret environment and names were requested. |
| Resources | Confirms worker count, CPU, memory, GPU, and services. |
| Refiner version | Confirms which package version workers used. |

## Debugging With Manifests

Use the manifest when:

- a cloud job behaved differently from a local run
- a teammate needs to reproduce the exact submission
- a dependency version changed between runs
- a secret was referenced from the wrong environment
- resource settings were too small or too expensive
- a resumed job should be compared to the original job

For billing investigations, open the job from the invoice breakdown, then open
the manifest to see worker count, GPU settings, and service configuration.

## Related Pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
- [Billing](billing.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)
