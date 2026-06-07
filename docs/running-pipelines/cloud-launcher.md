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

## Runtime Dependencies

Built-in Refiner readers, writers, and operations automatically declare the
[optional dependency groups](../reference/optional-dependencies.md) they need.
For example, `read_hf_dataset(...)` adds `datasets`, Hugging Face paths add
`hf`, HDF5 readers add `hdf5`, cloud storage paths add the relevant storage
extra, and `mdr.robotics.track_hands(...)` adds `hand_tracking`.

You can still pass `refiner_extras` explicitly when code outside the built-in
pipeline blocks needs a specific Refiner extra:

```python
pipeline.launch_cloud(
    name="custom-datasets-helper",
    refiner_extras=["datasets"],
)
```

For packages outside Refiner's optional feature groups, pass `dependencies`.
Each entry is a pip requirement string. Versionless requirements, exact pins,
ranges, and package extras are accepted:

```python
pipeline.launch_cloud(
    name="custom-model-job",
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
