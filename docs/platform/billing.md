---
title: "Billing"
description: "Understand workspace credits, payment state, usage, and invoice breakdowns"
---

# Billing

Billing is workspace-scoped. Jobs, worker compute, and runtime services are
attributed to the workspace that owns the submitted job.

Open [Settings > Billing](/settings/billing).

Only workspace owners and admins can manage payment details. Members can use
billing visibility if they have access to the workspace settings page, but
payment actions are restricted.

## Billing Page

The billing page shows:

| Area | What it means |
| --- | --- |
| Billing Period Spend | Current usage for the selected billing period. |
| Billing cycle selector | Prior and current billing periods when more than one is available. |
| Included or remaining credits | Credits available in the billing period. |
| Additional usage | Usage beyond included credits. |
| Payment Information | Card/setup state and Stripe actions. |
| Invoice Breakdown | Usage grouped by job, with worker and service rows. |

The invoice breakdown links each job line back to the job detail page when the
job is known. Use that link to explain where spend came from: the job graph,
worker count, service usage, logs, metrics, and manifest are all on the job
detail page.

## Credits And Additional Usage

The platform separates usage into included credits and additional usage.

| Term | Meaning |
| --- | --- |
| Included credits | Monthly workspace credits included before paid overage. |
| Remaining credits | Unused credits left in the active billing period. |
| Billing period spend | Total usage in the selected period. |
| Additional usage | Spend above included credits. |

Adding a card raises monthly credits to $25 and lets jobs keep running after
included credits run out. Without a usable card, cloud jobs can be blocked when
credits are exhausted.

## Payment States

Workspace cloud execution is gated by payment state and available credits.

| State | What it means | What to do |
| --- | --- | --- |
| `free_only` | No card is configured. The workspace can use included credits only. | Add a card before credits run out. |
| `paid_allowed` | A card is configured and paid usage is allowed. | Monitor current-period spend. |
| `payment_required` | The workspace needs a card to continue paid usage. | Click **Add Card** on the billing page. |
| `delinquent` | A previous payment failed. New cloud jobs are blocked. | Click **Manage Billing Details** and resolve payment in Stripe. |

The payment panel explains the current state. Depending on state and role, it
shows **Add Card** and/or **Manage Billing Details**.

## Add Or Manage A Card

1. Open [Settings > Billing](/settings/billing).
2. In **Payment Information**, click **Add Card** if no card is configured.
3. Complete the Stripe setup flow.
4. Return to the billing page.
5. Use **Manage Billing Details** later to update card, invoice, or billing
   information through Stripe.

If the page says payment services are unavailable, billing provider setup is not
configured for the deployment. Contact the Macrodata team.

## Submission Rules

Cloud job submission checks billing before work starts. Submission can be
blocked when:

- the workspace has no remaining included credits
- the workspace needs a payment method for additional usage
- the workspace is delinquent
- billing is unavailable for the workspace
- paid usage has reached the billing-period cap

The current paid-usage cap is $200 of paid overage for the billing period. A
workspace that reaches that cap cannot submit more paid jobs until the next
period or until the account is adjusted.

## Invoice Breakdown

The invoice breakdown groups usage by job. Each job group can include:

| Row kind | Meaning |
| --- | --- |
| Jobs / Compute Usage | Worker compute usage for the job. |
| Services | Runtime service usage, such as model-serving processes started for the job. |
| Unattributed usage | Usage that cannot be matched to a known job record. |

Use the arrow on a job row to open the job detail page. If a service row is
large, open the job and the [Services](services.md) page to see the service
instantiations and logs.

## Billing Cycles

When prior periods are available, the billing page shows a billing cycle
selector. Pick a period to see spend and invoice breakdown for that period.

Active periods can change while jobs run. Closed periods reflect the finalized
usage reported by the billing provider.

## Services And Billing

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
3. Check **Payment Information** for `payment_required` or `delinquent` copy.
4. Add or update the card through Stripe.
5. Retry the launch.

If spend looks surprising:

1. Open the invoice breakdown.
2. Expand the highest-cost job.
3. Open that job.
4. Check worker count, runtime services, duration, logs, metrics, and manifest.

## Related Pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Cloud Jobs and Files](cloud-jobs-and-files.md)
- [Services](services.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)
