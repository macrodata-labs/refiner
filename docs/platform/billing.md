---
title: "Billing"
description: "Understand workspace credits, payment methods, usage, and invoice breakdowns"
---

# Billing

Billing is workspace-scoped. Jobs, worker compute, and runtime services are
attributed to the workspace that owns the submitted job.

Open [Settings > Billing](/settings/billing).

Only workspace owners and admins can manage payment details. Members can use
billing visibility if they have access to the workspace settings page, but
payment actions are restricted.

## Billing page

The billing page shows:

| Area | What it means |
| --- | --- |
| Billing period spend | Current usage for the selected billing period. |
| Billing cycle selector | Prior and current billing periods when more than one is available. |
| Included or remaining credits | Credits available in the billing period. |
| Additional usage | Usage beyond included credits. |
| Payment information | Card status and Stripe actions. |
| Invoice breakdown | Usage grouped by Job, with worker and service rows. |

The invoice breakdown links each job line back to the job detail page when the
job is known. Use that link to explain where spend came from: the job graph,
worker count, service usage, logs, metrics, and manifest are all on the job
detail page.

## Credits and additional usage

The platform separates usage into included credits and additional usage.

| Term | Meaning |
| --- | --- |
| Included credits | Monthly workspace credits included before paid overage. |
| Remaining credits | Unused credits left in the active billing period. |
| Billing period spend | Total usage in the selected period. |
| Additional usage | Spend above included credits. |

Adding a card raises monthly credits to $30 and lets Jobs keep running after
included credits run out. Without a usable card, cloud jobs can be blocked when
credits are exhausted.

## Payment status

Workspace cloud execution depends on available credits and payment status.

| What you see | What it means | What to do |
| --- | --- | --- |
| No card is configured | The workspace can use included credits only. | Add a card before credits run out. |
| Card is configured | Jobs can continue into paid usage after included credits run out. | Monitor current-period spend. |
| Payment method required | The workspace needs a card before it can continue paid usage. | Click **Add Card** on the billing page. |
| Payment overdue | An overdue payment needs attention before new paid Jobs can run. | Click **Manage Billing Details** and resolve the issue in Stripe. |

The payment panel explains the current status. Depending on status and role, it
shows **Add Card** and/or **Manage Billing Details**.

## Add or manage a card

1. Open [Settings > Billing](/settings/billing).
2. In **Payment Information**, click **Add Card** if no card is configured.
3. Complete the Stripe setup flow.
4. Return to the billing page.
5. Use **Manage Billing Details** later to update card, invoice, or billing
   information through Stripe.

## Submission rules

Cloud Job submission checks billing before work starts. Submission can be
blocked when:

- the workspace has no remaining included credits
- the workspace needs a payment method for additional usage
- a payment is overdue
- workspace paid usage has reached its billing-period limit

When a workspace reaches its billing-period limit, new paid Jobs wait until the
next period or until the account is adjusted.

## Invoice breakdown

The invoice breakdown groups usage by job. Each job group can include:

| Row kind | Meaning |
| --- | --- |
| Jobs / Compute Usage | Worker compute usage for the job. |
| Services | Runtime service usage, such as model-serving processes started for the job. |
| Unattributed usage | Usage that cannot be matched to a known job record. |

Use the arrow on a job row to open the job detail page. If a service row is
large, open the job and the [Services](services.md) page to see the service
instantiations and logs.

## Billing cycles

When prior periods are available, the billing page shows a billing cycle
selector. Pick a period to see spend and invoice breakdown for that period.

Active periods can change while jobs run. Closed periods reflect the finalized
usage reported by the billing provider.

## Services and billing

Runtime services are billed separately from worker compute but remain grouped
under the job that started them when possible. This matters for inference-heavy
robotics pipelines where workers call a shared vLLM service: the job line shows
compute usage and service usage together.

Open [Services](/services) to inspect running and stopped service groups. See
[Services](services.md).

## Troubleshooting

If jobs stop submitting:

1. Open [Settings > Billing](/settings/billing).
2. Check whether credits are exhausted.
3. Check **Payment Information** for card setup or overdue payment copy.
4. Add or update the card through Stripe.
5. Retry the launch.

If spend looks surprising:

1. Open the invoice breakdown.
2. Expand the highest-cost job.
3. Open that job.
4. Check worker count, runtime services, duration, logs, metrics, and manifest.

## Related pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Cloud Jobs and Files](cloud-jobs-and-files.md)
- [Services](services.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)
