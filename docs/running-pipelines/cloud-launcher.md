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
    refiner_extras=["hand_tracking"],
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

## Runtime Dependencies

By default, cloud launch installs base Refiner. Add only the runtime pieces your
workers need.

If your pipeline needs a Refiner feature such as Hugging Face readers, HDF5,
video, S3, or hand tracking, pass the matching `refiner_extras`. See the
[optional dependency groups](../reference/optional-dependencies.md):

```python
pipeline.launch_cloud(
    name="hand-tracking",
    gpu=mdr.GPU(count=1, type="h100", cuda_version="12.8"),
    refiner_extras=["hand_tracking"],
)
```

For packages outside Refiner's optional feature groups, pass `dependencies`.
Each entry is a pip requirement string. Versionless requirements, exact pins,
ranges, and package extras are accepted:

```python
pipeline.launch_cloud(
    name="custom-model-job",
    refiner_extras=["hf"],
    dependencies=[
        "torch",
        "transformers>=4.55",
    ],
)
```

Environment markers are not preserved. Do not include markers in
`dependencies`; list the package as it should install on Macrodata Cloud.
For example, write `uvloop`, not `uvloop; sys_platform != "win32"`.

Finally, if you would like Refiner to try syncing the packages installed in
your current Python environment, set `sync_local_dependencies=True`. Explicit
`dependencies` take precedence over synced packages with the same package name.
If any synced package cannot be resolved from PyPI, the cloud job setup will
fail.

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
