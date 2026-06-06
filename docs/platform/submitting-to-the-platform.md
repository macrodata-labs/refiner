---
title: "Submitting to the Platform"
description: "Authenticate, launch cloud jobs, and inspect submitted Refiner runs"
---

# Submitting To The Platform

Submitting a Refiner pipeline to Macrodata Cloud creates a workspace-owned job
record. The platform stores the run metadata, starts workers, tracks progress,
collects logs and metrics, preserves the manifest, and exposes the job in the
browser and CLI.

## Before You Submit

You need:

| Requirement | Where to set it up |
| --- | --- |
| Workspace | [Settings > Workspaces](/settings/workspaces) |
| API key | [Settings > API Keys](/settings/api-keys) |
| Secrets for private data or model providers | [Settings > Secrets](/settings/secrets) |
| Billing capacity for cloud execution | [Settings > Billing](/settings/billing) |

Run locally first when possible. Cloud launch should be the step after your
reader, transforms, schemas, and writer work on a small local sample.

## Authenticate

Create an API key in [Settings > API Keys](/settings/api-keys), then log in:

```bash
macrodata login --token md_...
macrodata whoami
```

`whoami` confirms the user, key name, and workspace:

```text
Logged in as Ada Lovelace <ada@example.com>
API key name: laptop
Workspace: Acme Robotics (acme-robotics)
```

For CI or agent runs, set the environment variable instead of storing local
credentials:

```bash
export MACRODATA_API_KEY="md_..."
macrodata whoami
```

## Launch A Cloud Job

Use `launch_cloud()` on the same pipeline object you would launch locally:

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("lerobot/aloha_mobile_cabinet")
    .map(score_episode)
    .write_parquet("s3://acme-robotics/datasets/aloha-scored")
)

pipeline.launch_cloud(
    name="aloha-scoring",
    num_workers=16,
    cpus_per_worker=8,
    mem_mb_per_worker=32768,
    refiner_extras=["hf", "s3"],
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

Common launch settings:

| Setting | Meaning |
| --- | --- |
| `name` | Display name in [Jobs](/jobs). |
| `num_workers` | Number of workers that claim shards. |
| `cpus_per_worker` | CPU allocation per worker. |
| `mem_mb_per_worker` | Memory allocation per worker. |
| `gpu` | GPU type/count per worker. |
| `refiner_extras` | Refiner [optional dependency groups](../reference/optional-dependencies.md) to install on workers, such as `["hf", "video"]`. |
| `dependencies` | Additional pip requirement strings to install on workers. |
| `sync_local_dependencies` | Ask Refiner to try syncing packages from the submitting environment. Defaults to `False`. |
| `secrets` | Secret values or workspace secret references passed to workers. |
| `env` | Non-secret environment variables. |
| `continue_from_job` | Reuse compatible outputs from a prior cloud job. |

See [Cloud Launcher](../running-pipelines/cloud-launcher.md) and
[Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md).

### Dependency Overrides

If your job requires support for a Refiner feature, pass the matching
`refiner_extras`. See the
[optional dependency groups](../reference/optional-dependencies.md):

```python
pipeline.launch_cloud(
    name="hand-tracking",
    gpu=mdr.GPU(count=1, type="h100", cuda_version="12.8"),
    refiner_extras=["hand_tracking"],
)
```

If your code needs other packages, pass them with `dependencies`:

```python
pipeline.launch_cloud(
    name="custom-model-job",
    refiner_extras=["hf"],
    dependencies=["torch", "transformers>=4.55"],
)
```

Dependency entries are pip requirement strings. Exact pins, versionless
requirements, ranges, and package extras are supported.

Environment markers are not preserved. Do not include markers in
`dependencies`; list the package as it should install on Macrodata Cloud.
For example, write `uvloop`, not `uvloop; sys_platform != "win32"`.

Finally, if you would like Refiner to try syncing the packages installed in
your current Python environment, set `sync_local_dependencies=True`. Explicit
`dependencies` take precedence over synced packages with the same package name.
If one of those packages cannot be resolved from PyPI during cloud image setup,
the job will fail.

## What Gets Submitted

A cloud submission includes:

| Item | Why it matters |
| --- | --- |
| Pipeline plan | Stages, reader, transforms, writer, sharding, and dependencies between steps. |
| Code snapshot | The submitted Python code workers execute. |
| Dependency manifest | The Python/runtime package context captured for reproduction. |
| Secret references | Names and environments, not secret values. |
| Resource plan | Worker count, CPU, memory, GPU, and runtime services. |

The manifest is available in the job page's **Manifest** tab and through:

```bash
macrodata jobs manifest job_123 --deps --code
```

See [Manifests](manifests.md).

## Find The Job

After submission, open [Jobs](/jobs). The jobs list shows name, status,
progress, duration, starter, and an action to open the job. Use **Load more**
when older jobs are paginated.

From the CLI:

```bash
macrodata jobs list
macrodata jobs get job_123
```

Use `--json` when another program needs to parse the result:

```bash
macrodata jobs get job_123 --json
```

## Inspect A Job

Open the job from [Jobs](/jobs). The job detail page has:

| Area | What it shows |
| --- | --- |
| Header | Job name, ID, status, executor kind, attempts, and cancel action when available. |
| Overview | Pipeline graph with stages and steps. |
| Stage panel | Stage status, shard counts, and selected-stage detail. |
| Worker observability | Workers, logs, metrics, and resource metrics for cloud jobs. |
| Manifest tab | Captured code, dependencies, and runtime metadata. |
| Job info sidebar | Timestamps, cost, owner, lineage, and other metadata. |

Cloud jobs stream progress and observability while they run. Local jobs can
appear in the platform, but advanced real-time observability is for cloud jobs.

## Logs And Metrics

Use the job page for interactive debugging. Use the CLI for terminal workflows:

```bash
macrodata jobs logs job_123 --follow
macrodata jobs logs job_123 --stage 0 --severity error
macrodata jobs metrics job_123 0
macrodata jobs metrics job_123 0 --step 2 --metric rows_processed --workers
macrodata jobs resource-metrics job_123 0 --range 15m
```

See [Observability](../running-pipelines/observability.md) and
[CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md).

## Cancel A Job

Cancel from the job detail page when the header exposes cancellation, or use:

```bash
macrodata jobs cancel job_123
```

Cancellation requests stop active work for that job. Use `jobs get` afterward
to confirm the terminal state.

## Billing And Submission Gates

Cloud submission checks workspace billing state. A workspace can submit while
it has available included credits, or after a payment method is configured for
paid usage.

Submissions are blocked when:

- billing is not configured for the workspace
- the workspace is delinquent
- a workspace without a usable payment method has exhausted included credits
- the workspace reaches the paid-usage cap for the billing period

Open [Settings > Billing](/settings/billing) to add a card, manage billing
details, inspect usage, or choose a prior billing cycle. See [Billing](billing.md).

## Related Pages

- [Workspaces and API Keys](workspaces-and-api-keys.md)
- [Secrets and Environment](secrets-and-environment.md)
- [Cloud Jobs and Files](cloud-jobs-and-files.md)
- [CLI](../cli/index.md)
