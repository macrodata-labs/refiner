---
title: "Platform"
description: "Use Macrodata Cloud to run, track, share, and inspect Refiner work"
---

# Platform

Macrodata Cloud turns Refiner pipelines from local scripts into shared robotics
data operations. The code stays Python, but the run gets durable execution,
workspace ownership, a job record, logs, metrics, secrets, manifests, files,
runtime services, billing visibility, and a browser viewer for produced data.

Use it when a laptop run becomes team infrastructure: converting robot episodes
at scale, running GPU annotation, refreshing datasets, checking failed shards,
or handing a run to another engineer without losing the execution context.

## Platform Overview

The platform keeps track of every job submitted to a workspace. Each job page
keeps the run inspectable after submission:

- what code and pipeline plan was submitted
- who submitted it and when it started
- which stages, workers, and shards are running
- where logs, metrics, services, manifests, and output references live
- whether the run is queued, running, succeeded, failed, or canceled

Open the jobs list from [Jobs](/jobs). Clicking a job opens its detail page,
where the graph, stage panels, worker observability, logs, metrics, and
manifest live.

## Workspaces

Workspaces organize jobs, API keys, secrets, members, billing, services, and
viewer access. Everyone starts with a personal workspace. Personal workspaces
are useful for solo experiments and first jobs.

Create additional workspaces when the work belongs to a team, project, lab, or
production dataset. Shared workspaces can be invited to other users, and their
jobs, secrets, API keys, billing, services, and viewer access stay scoped to
that workspace.

To manage workspaces:

1. Open [Settings > Workspaces](/settings/workspaces).
2. Click **Create Workspace**.
3. Enter a workspace name and slug.
4. Use **Switch** to make a workspace active.
5. Use **Manage Members** on a shared workspace, or open
   [Settings > Members](/settings/members).

Workspace-specific settings are under:

| Screen | URL |
| --- | --- |
| General | [Settings > General](/settings/general) |
| Billing | [Settings > Billing](/settings/billing) |
| API Keys | [Settings > API Keys](/settings/api-keys) |
| Secrets | [Settings > Secrets](/settings/secrets) |
| Members | [Settings > Members](/settings/members) |

## Submitting to the Platform

Cloud submission starts with an API key scoped to the workspace that should own
the job. Create one from [Settings > API Keys](/settings/api-keys), copy it
once, then authenticate locally:

```bash
macrodata login --token md_...
macrodata whoami
```

Then submit the same pipeline you debugged locally:

```python
pipeline.launch_cloud(
    name="aloha-reward-scoring",
    num_workers=16,
    cpus_per_worker=8,
    mem_mb_per_worker=32768,
    refiner_extras=["hf", "video"],
    secrets=mdr.Secrets.env(name="production", keys=["HF_TOKEN"]),
)
```

The submitted job appears in [Jobs](/jobs). Use the job detail page for the
graph, logs, metrics, workers, manifest, and cancellation. Use the CLI when an
agent or script needs structured output:

```bash
macrodata jobs list --json
macrodata jobs get job_123 --json
macrodata jobs logs job_123 --follow
```

See [Submitting to the Platform](submitting-to-the-platform.md),
[Cloud Launcher](../running-pipelines/cloud-launcher.md), and
[CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md).

## Billing

Billing is workspace-scoped. Open [Settings > Billing](/settings/billing) to
see current billing-period spend, included or remaining credits, additional
usage, payment method actions, prior billing cycles, and an invoice breakdown
grouped by job.

Usage is attributed to the workspace that owns the job. Worker compute and
runtime services both show up in the invoice breakdown, so a robotics
annotation run can be traced from billing line item back to the job that
started it.

Payment state affects cloud execution. A workspace with remaining included
credits can submit jobs. Adding a card raises monthly credits to $25 and allows
paid usage after included credits run out. New paid jobs are blocked when a
workspace needs a payment method, is delinquent, or hits the paid-usage cap for
the billing period.

See [Billing](billing.md).

## Services

Services are runtime processes started for jobs, usually for model serving
workloads such as vLLM-backed inference. They let workers call a shared service
instead of loading an expensive model in every worker process.

Open [Services](/services) to see running and stopped service groups. Click
**View** to inspect instantiations, service logs, the source job, stage index,
duration, model, and configuration label.

See [Services](services.md) and
[Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md).

## Viewer

The viewer opens Parquet, JSON, and CSV files directly in the browser from S3,
GCS, Hugging Face, HTTP, or HTTPS paths. It uses workspace secrets to resolve
private storage into browser-fetchable URLs, then DuckDB-Wasm queries the file
client-side for preview, search, sort, pagination, and media cells.

Open [Viewer](/viewer), choose a secrets environment, paste a file path, choose
a file type or **Auto**, and click **Load**. Use
[Settings > Secrets](/settings/secrets) first for private buckets or private
Hugging Face repos.

See [Viewer](viewer.md).

## Platform Docs

| Page | Use it for |
| --- | --- |
| [Workspaces and API Keys](workspaces-and-api-keys.md) | Workspace organization, members, roles, and API key setup. |
| [Submitting to the Platform](submitting-to-the-platform.md) | Cloud launch, API-key auth, job ownership, and where submitted jobs appear. |
| [Billing](billing.md) | Credits, payment state, invoice breakdown, caps, and Stripe actions. |
| [Services](services.md) | Runtime service groups, instantiations, logs, and job links. |
| [Viewer](viewer.md) | Inspecting Parquet, JSON, CSV, media cells, and private storage. |
| [Manifests](manifests.md) | Reproducing exactly what a cloud job ran. |
| [Secrets and Environment](secrets-and-environment.md) | Passing credentials and configuration to jobs and the viewer. |
| [Cloud Jobs and Files](cloud-jobs-and-files.md) | Job lifecycle and file staging behavior. |
