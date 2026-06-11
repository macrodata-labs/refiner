---
title: "Manifests"
description: "Understand and inspect Refiner cloud job manifests"
---

# Manifests

A manifest records what Refiner submitted for a cloud job. It is the
reproducibility record for a run: code, dependency context, pipeline plan,
resource settings, and secret references.

Open a job from [Jobs](/jobs), then click the **Manifest** tab.

## What a manifest answers

Use a manifest to answer:

- what code was submitted
- what Python/runtime dependencies were captured
- which pipeline plan ran
- which reader, transforms, writer, stages, and resource settings were used
- which secret names and environments were referenced
- which Refiner version and runtime context workers saw

Manifest records should not contain secret values.

## Inspect from the CLI

```bash
macrodata jobs manifest job_123
macrodata jobs manifest job_123 --deps
macrodata jobs manifest job_123 --code
macrodata jobs manifest job_123 --json
```

Use `--deps` to show dependencies and `--code` to show captured script text.
Use `--json` for an agent, notebook, or CI job.

## Dependency entries

Built-in Refiner blocks automatically add the
[optional dependency groups](../reference/optional-dependencies.md) they need to
the manifest. Hugging Face dataset readers, Hugging Face paths, HDF5/Zarr
readers, video writers, cloud storage paths, and hand tracking operations are
recorded from the pipeline plan.

Use `refiner_extras` only when code outside the built-in blocks needs a specific
Refiner extra:

```python
pipeline.launch_cloud(
    name="custom-datasets-helper",
    refiner_extras=["datasets"],
)
```

Use `dependencies` for other packages needed by your code:

```python
pipeline.launch_cloud(
    name="custom-model-job",
    dependencies=[
        "torch==2.6.0",
        "transformers>=4.55",
    ],
)
```

Exact pins are shown as package/version pairs. Versionless requirements and
ranges are shown as the submitted install string:

```text
torch==2.6.0
transformers>=4.55
```

Environment markers are not preserved. Do not include markers in
`dependencies`; list the package as it should install on the Macrodata Cloud.
For example, write `uvloop`, not `uvloop; sys_platform != "win32"`.

Finally, if `sync_local_dependencies=True`, Refiner tries to sync packages from
the submitting Python environment. Those entries appear as package/version
pairs in the manifest. Explicit `dependencies` take precedence over synced
packages with the same package name. If any synced package cannot be resolved
from PyPI during cloud image setup, the job will fail.

## What to look for

| Field | Why it matters |
| --- | --- |
| Pipeline plan | Confirms the reader, stages, transforms, and writer. |
| Dependencies | Confirms model/runtime libraries available to workers. |
| Code | Confirms the submitted script or module. |
| Secret references | Confirms which workspace secret environment and names were requested. |
| Resources | Confirms worker count, CPU, memory, GPU, and services. |
| Refiner version | Confirms which package version workers used. |

## Debugging with manifests

Use the manifest when:

- a cloud job behaved differently from a local run
- a teammate needs to reproduce the exact submission
- a dependency version changed between runs
- a secret was referenced from the wrong environment
- resource settings were too small or too expensive
- a resumed job should be compared to the original job

For billing investigations, open the job from the usage breakdown, then open
the manifest to see worker count, GPU settings, and service configuration.

## Related pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
- [Billing](billing.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)
