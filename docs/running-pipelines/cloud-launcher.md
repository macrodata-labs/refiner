---
title: "Cloud launcher"
description: "Submit Refiner pipelines to the Macrodata Cloud"
---

# Cloud launcher

> [!WARNING]
>
> **Early access:** Macrodata Cloud is currently available to approved
> partners. Refiner remains open source and runs locally without platform
> access. [Contact Macrodata](/contact) to discuss access.

Cloud launch submits a pipeline to the Macrodata Cloud, where workers claim
shards, run transforms, write outputs, and report logs and metrics.

```python
pipeline.launch_cloud(
    name="aloha-trim",
    num_workers="auto",
    cpus_per_worker=4,
    mem_mb_per_worker=8192,
    secrets={"HF_TOKEN": None},
)
```

Set `num_workers="auto"` to request one worker for every shard in each stage.
The cloud registration runtime discovers shards after secrets and environment
variables are mounted, then Macrodata Cloud applies its normal worker and GPU
limits to the resulting count. The submitting machine does not need access to
private inputs. An empty stage starts no worker containers and completes after
shard registration. Pass a positive integer when you want a fixed worker count.

## Cloud and region placement

Workers use AWS by default. Select one supported public cloud with `cloud`:

```python
pipeline.launch_cloud(
    name="gcp-workers",
    cloud="gcp",  # "aws", "oci", or "gcp"
)
```

By default, workers may run in the US, EEA, or Canada. Pass one selector or a
list; a worker is accepted when it matches any selector:

```python
pipeline.launch_cloud(
    name="north-america-workers",
    cloud="aws",
    region=["us-east", "ca"],
)
```

Broad selectors are `us`, `eu`, `ca`, and `uk`. Narrow selectors are
`us-east`, `us-central`, `us-south`, `us-west`, `eu-west`, `eu-north`, and
`eu-south`. `eu` excludes the UK. Madrid is classified as `eu-south`; the
defensive `FRA*` and `AMS` aliases are classified as `eu-west`.

The default `placement_mode="best_effort"` lets the provider choose a region,
then rejects and safely retries a worker that lands outside the requested
selectors. Use strict placement when a job must not spill to another region:

```python
pipeline.launch_cloud(
    name="strict-eu-workers",
    cloud="aws",
    region=["eu-west", "uk"],
    placement_mode="strict",
)
```

Strict placement also sends the region list as a native provider placement
constraint. If the provider cannot satisfy it, the job remains fail-closed
instead of silently running elsewhere.

## What gets submitted

A cloud submission includes:

| Item | User-visible purpose |
| --- | --- |
| Pipeline plan | Describes the reader, transforms, writer, and stages. |
| Code snapshot | Lets workers run the same code you submitted. |
| Dependency manifest | Helps reproduce the Python environment. |
| Secrets/env mapping | Supplies credentials and runtime configuration without hard-coding them in code. |

You can inspect submitted metadata through the platform and CLI. See
[Manifests](../platform/manifests.md) and [CLI Jobs](../cli/jobs-logs-and-metrics.md).

## Runtime dependencies

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
`dependencies`; list the package as it should install on the Macrodata Cloud.
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

See [Secrets and environment](../platform/secrets-and-environment.md).

## Continue from a prior Job

Use continuation when earlier stages already produced reusable outputs:

```python
pipeline.launch_cloud(
    name="resume-after-fix",
    continue_from_job="infer",
)
```

`"infer"` asks Refiner to find a compatible prior job. Use explicit job IDs when
you need deterministic behavior.

## Related pages

- [Resources, GPUs, and Services](resources-gpus-and-services.md)
- [Observability](observability.md)
- [Platform Jobs and Files](../platform/cloud-jobs-and-files.md)
