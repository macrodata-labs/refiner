---
title: "Cloud Launcher"
description: "Submit Refiner pipelines to Macrodata Cloud"
---

# Cloud Launcher

Cloud launch submits a pipeline to Macrodata Cloud, where workers claim shards,
run transforms, write outputs, and report logs and metrics.

```python
pipeline.launch_cloud(
    name="aloha-trim",
    num_workers=8,
    cpus_per_worker=4,
    mem_mb_per_worker=8192,
    secrets={"HF_TOKEN": None},
)
```

## What Gets Submitted

A cloud submission includes:

| Item | User-visible purpose |
| --- | --- |
| Pipeline plan | Describes the reader, transforms, writer, and stages. |
| Code snapshot | Lets workers run the same code you submitted. |
| Dependency manifest | Helps reproduce the Python environment. |
| Secrets/env mapping | Supplies credentials without hard-coding them in code. |

You can inspect submitted metadata through the platform and CLI. See
[Manifests](../platform/manifests.md) and [CLI Jobs](../cli/jobs-logs-and-metrics.md).

## Secrets

For local environment values:

```python
pipeline.launch_cloud(
    name="with-env-secret",
    secrets={"HF_TOKEN": None},
)
```

For stored workspace secrets:

```python
pipeline.launch_cloud(
    name="with-stored-secret",
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

See [Secrets and Environment](../platform/secrets-and-environment.md).

## Continue From A Prior Job

Use continuation when earlier stages already produced reusable outputs:

```python
pipeline.launch_cloud(
    name="resume-after-fix",
    continue_from_job="infer",
)
```

`"infer"` asks Refiner to find a compatible prior job. Use explicit job IDs when
you need deterministic behavior.

## Related Pages

- [Resources, GPUs, and Services](resources-gpus-and-services.md)
- [Observability](observability.md)
- [Platform Jobs and Files](../platform/cloud-jobs-and-files.md)
