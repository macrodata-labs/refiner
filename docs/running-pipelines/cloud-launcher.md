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
    extra_dependencies=["macrodata-refiner[hand_tracking]"],
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

By default, cloud launch captures packages from the local Python environment and
adds them to the job manifest. Workers install the submitted dependency list
before running the pipeline.

Use `extra_dependencies` when a cloud worker needs packages that are not present
locally, or when the cloud package should override the locally captured version:

```python
pipeline.launch_cloud(
    name="hand-tracking",
    gpu=mdr.GPU(count=1, type="h100", cuda_version="12.8"),
    extra_dependencies=["macrodata-refiner[hand_tracking]"],
)
```

Each entry is a pip requirement string. Versionless requirements, exact pins,
ranges, and extras are accepted:

```python
extra_dependencies=[
    "torch",
    "transformers>=4.55",
    "macrodata-refiner[hand_tracking]",
]
```

Environment markers are not preserved. Do not include markers in
`extra_dependencies`; list the package as it should install on Macrodata Cloud.
For example, write `uvloop`, not `uvloop; sys_platform != "win32"`.

Extra dependencies take precedence over captured local packages with the same
package name. For example, if the local environment has `torch==2.4.0` but
`extra_dependencies` includes `torch==2.6.0`, the submitted manifest keeps the
explicit `torch==2.6.0` pin.

Set `sync_local_dependencies=False` to skip local environment capture. Extra
dependencies still install:

```python
pipeline.launch_cloud(
    name="minimal-runtime",
    sync_local_dependencies=False,
    extra_dependencies=["torch", "macrodata-refiner[hf]"],
)
```

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
