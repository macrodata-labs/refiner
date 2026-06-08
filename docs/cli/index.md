---
title: "CLI"
description: "Use the macrodata command line interface with Refiner"
---

# CLI

The `macrodata` CLI authenticates, submits scripts, inspects jobs, follows logs,
reads metrics, and manages secrets.

Most inspection commands have a human-readable default and a `--json` mode for
agents, scripts, notebooks, and CI jobs. Prefer `--json` whenever another
program will parse the output.

```bash
macrodata jobs get job_123 --json
macrodata jobs logs job_123 --stage 0 --severity error --json
macrodata secrets list --env production --json
```

## Subcommands

- [`login`](auth-and-run.md#login): Store and validate a Macrodata API key.
- [`whoami`](auth-and-run.md#check-identity): Validate local credentials and show identity.
- [`logout`](auth-and-run.md#logout): Remove local Macrodata credentials.
- [`run`](auth-and-run.md#run-a-script): Run a Macrodata Refiner pipeline script.
- [`jobs list`](jobs-logs-and-metrics.md#list-jobs): List jobs in the current workspace.
- [`jobs get`](jobs-logs-and-metrics.md#inspect-one-job): Get job summary.
- [`jobs attach`](jobs-logs-and-metrics.md#attach-to-a-running-job): Attach to a running cloud job.
- [`jobs manifest`](jobs-logs-and-metrics.md#manifest): Get job manifest.
- [`jobs workers`](jobs-logs-and-metrics.md#workers): List job workers.
- [`jobs logs`](jobs-logs-and-metrics.md#follow-logs): Fetch cloud job logs.
- [`jobs metrics`](jobs-logs-and-metrics.md#stage-metrics): Fetch cloud step metrics for a stage.
- [`jobs resource-metrics`](jobs-logs-and-metrics.md#resource-metrics): Fetch cloud resource metrics for a stage.
- [`jobs cancel`](jobs-logs-and-metrics.md#cancel): Cancel a cloud job.
- [`secrets list`](secrets.md#list-secrets): List workspace secret names.
- [`secrets set`](secrets.md#set-a-secret): Add or replace a workspace secret.
- [`secrets remove`](secrets.md#remove-a-secret): Remove a workspace secret.
- [`secrets delete`](secrets.md#remove-a-secret): Alias for `secrets remove`.

## Agent-friendly output

Use `--json` on commands that return structured platform data:

- `macrodata jobs list --json`
- `macrodata jobs get <job_id> --json`
- `macrodata jobs manifest <job_id> --json`
- `macrodata jobs workers <job_id> --json`
- `macrodata jobs logs <job_id> --json`
- `macrodata jobs metrics <job_id> <stage_index> --json`
- `macrodata jobs resource-metrics <job_id> <stage_index> --json`
- `macrodata jobs cancel <job_id> --json`
- `macrodata secrets list --json`
- `macrodata secrets set <name> --value-stdin --json`
- `macrodata secrets remove <name> --json`
- `macrodata secrets delete <name> --json`

Pagination commands return cursors in JSON responses and print a follow-up
command in human-readable mode. Use `--cursor` with `jobs list`, `jobs workers`,
and `jobs logs`.

## Help

Every command exposes its current options through `--help`:

```bash
macrodata --help
macrodata jobs logs --help
macrodata jobs metrics --help
```
