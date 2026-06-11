---
title: "Billing"
description: "Understand workspace credits, payment methods, usage, and auto-recharge"
---

# Billing

Macrodata Cloud runs on credits. Cloud jobs spend workspace credits as they
run, based on the CPU, memory, GPU time, and runtime services they use.
Macrodata measures that usage, applies the published pricing rates, and
subtracts the result from the workspace credit balance.

If a workspace runs out of credits, running cloud jobs are canceled and new
cloud jobs cannot be submitted until credits are restored. To help avoid
interruptions, workspace owners and admins can enable auto-recharge after
saving a payment method.

Open [Settings > Billing](/settings/billing).

Only workspace owners and admins can manage payment details. Members can use
billing visibility if they have access to the workspace settings page, but
payment actions are restricted.

## How credits work

Every workspace gets **$10 in free credits each month**. After you save a
payment method, the monthly free credit amount increases to **$30**.

If you run out of free credits, you can either buy more credits manually or
enable auto-recharge to add credits automatically before the balance reaches
zero.

Cloud usage spends credits as it runs. Worker CPU, memory, GPU time, and
runtime services are measured per second, converted to credits using the
published resource rates, and subtracted from the workspace balance. See
[pricing](/pricing) for the current rates.

Purchased credits become available after payment succeeds.

Local jobs do not spend workspace credits.

## What happens when credits run out

Macrodata Cloud checks the workspace credit balance before starting new cloud
work. If the workspace has no available credits, new cloud jobs are rejected
until credits are restored.

Running cloud jobs are canceled when the workspace credit balance reaches zero.
New cloud jobs remain blocked until the balance is restored with monthly
credits, purchased credits, or auto-recharge. Local jobs are not canceled by
cloud billing limits.

To avoid interruptions, workspace owners and admins can add prepaid credits
manually or configure auto-recharge after saving a payment method. Auto-recharge
automatically buys credits when the workspace balance falls below the threshold
you choose.

## Buy credits and auto-recharge

Use **Add to credit balance** to add a custom prepaid amount between $5 and
$85. The billing page shows the estimated total before payment confirmation,
and credits become available after the payment succeeds.

Auto-recharge can add prepaid credits automatically when a workspace balance
falls below a threshold:

1. Open **Setup auto recharge** or **Auto recharge settings**.
2. Turn on **Auto-Recharge**.
3. Set **When balance drops to**.
4. Set **Restore balance to**.
5. Click **Save**.

When the balance reaches the threshold, Macrodata charges the saved payment
method through Stripe and adds enough prepaid credits to restore the target
balance.

## Add or manage a card

1. Open [Settings > Billing](/settings/billing).
2. In **Payment Information**, click **Add Card** if no card is configured.
3. Complete the Stripe setup flow.
4. Return to the billing page.
5. Use **Manage Billing & Invoices** later to update card, invoice, or billing
   information through Stripe.

## Usage breakdown

The usage breakdown groups credit usage by job. Each job group can include:

| Row kind | Meaning |
| --- | --- |
| Jobs / Compute Usage | Worker compute usage for the job. |
| Services | Runtime service usage, such as model-serving processes started for the job. |

Use the arrow on a job row to open the job detail page. If a service row is
large, open the job and the [Services](services.md) page to see the service
instantiations and logs.

## Related pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [Cloud Jobs and Files](cloud-jobs-and-files.md)
- [Services](services.md)
- [CLI Jobs, Logs, and Metrics](../cli/jobs-logs-and-metrics.md)
