---
title: "Services"
description: "Inspect runtime services started by Refiner cloud jobs"
---

# Services

Services are runtime processes started for cloud jobs. They are used when a
pipeline needs a shared process alongside workers, most commonly model-serving
workloads such as vLLM-backed inference.

Open [Services](/services), which redirects to the active workspace.

## Why Services Exist

Some robotics data workflows need expensive model processes: VLM annotation,
reward scoring, captioning, embedding, or policy-evaluation helpers. Loading
that model inside every worker can waste GPU memory and startup time.

A runtime service lets Refiner include a managed service in the cloud runtime
plan. Workers then call the service while processing shards.

See [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
and [Providers and vLLM](../inference/providers-and-vllm.md).

## Services List

The services list is split into **Running** and **Stopped** sections.

Each row shows:

| Column | Meaning |
| --- | --- |
| Model | Model name when known, otherwise the service name. |
| Config | Service configuration label. |
| Running | Number of currently running instances. |
| Total | Total instances recorded for the service group. |
| Action | **View** opens the service detail page. |

If no job has started a runtime service in the workspace, the page says that no
services have been instantiated yet.

## Service Detail

Click **View** from [Services](/services).

The service detail page shows:

| Area | Meaning |
| --- | --- |
| Header | Service name, service kind, and model name when available. |
| Instantiations | Job/stage groups that started service instances. |
| Logs | Logs for the selected instantiation group. |
| Open job | Link back to the job detail page. |

Instantiations include the job name, stage index, instance count, duration, and
running/stopped status. Use them to connect service behavior back to the job
and stage that used the service.

## Logs

The logs panel shows service logs for the selected instantiation group. These
are separate from worker logs but use the same platform log viewer. Use service
logs to debug model-server startup, readiness, load failures, request errors,
or shutdown behavior.

The job detail logs panel can include service logs alongside worker logs. In
the job page, use the logs controls to include or hide service sources.

## Billing

Runtime service usage appears in the workspace billing breakdown as **Services**
rows under the job that started them when attribution is available.

To investigate service spend:

1. Open [Settings > Billing](/settings/billing).
2. Expand the job in **Invoice Breakdown**.
3. Look for **Services** rows.
4. Open the job from the invoice breakdown.
5. Open the service from [Services](/services) for service instantiations and
   logs.

See [Billing](billing.md).

## Related Pages

- [Resources, GPUs, and Services](../running-pipelines/resources-gpus-and-services.md)
- [Providers and vLLM](../inference/providers-and-vllm.md)
- [Observability](../running-pipelines/observability.md)
- [Billing](billing.md)
