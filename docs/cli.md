---
title: "CLI"
description: "Macrodata CLI commands used with Refiner"
---

Refiner installs the `macrodata` CLI and the shorter `mdr` alias. They are equivalent, so `mdr jobs list` and `macrodata jobs list` run the same command.

## Auth

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

### Login

Options:

- `--token <api_key>`
- `--token-stdin`

```bash
macrodata login
```

```bash
macrodata login --token md_xxx
printf '%s' 'md_xxx' | macrodata login --token-stdin
```

### Check Current Auth

```bash
macrodata whoami
```

### Logout

```bash
macrodata logout
```

## Run A Script

Use `macrodata run` to run a Macrodata Refiner pipeline script.

Options:

- `--attach`
- `--detach`
- `--logs all|none|one|errors`
- script arguments after `--`

Behavior:

- `Ctrl+C` exits with code `130`
- local launcher resume/failure messages are printed cleanly, while ordinary script exceptions still surface normally
- the script directory is added to `sys.path`, so sibling imports work the same way they do with `python script.py`
- `macrodata run` defaults cloud launches to attach in interactive terminals and detach in non-interactive output
- `--attach` forces attached mode for cloud launches and is accepted for local launches
- `--detach` forces detached mode for cloud launches and errors for local launches
- `--logs` applies to attached local and cloud launches; `all` is the default, `one` shows a single worker at a time, `none` keeps the header live without log lines, and `errors` only shows error log lines
- direct Python `launch_cloud(...)` calls remain detached by default unless `REFINER_ATTACH` is explicitly set
- detached cloud launches print the exact follow-up commands to inspect, attach, or cancel the job
- attached cloud launches exit automatically when the job reaches a terminal state
- `Ctrl+C` during an attached cloud launch detaches the local CLI only; the cloud job keeps running and the CLI prints the job URL plus reattach and cancel commands

```bash
macrodata run path/to/pipeline.py
```

```bash
macrodata run --logs one path/to/pipeline.py -- --workers 4 --rows 20
macrodata run --detach path/to/cloud_pipeline.py
macrodata run --attach path/to/cloud_pipeline.py
```

## Credential Lookup

See [Auth](auth.md) for the shared credential lookup order and credential file location.

## Job Inspection

Inspect jobs in the workspace attached to your current API key:

### `macrodata jobs list`

Lists jobs visible to the current API key. Use it to find recent jobs, narrow to cloud or local runs, and page through longer result sets.

Options:

- `--status <status>`
- `--kind local|cloud`
- `--limit <count>`
- `--me`
- `--cursor <cursor>`
- `--json`

```bash
macrodata jobs list --kind cloud
macrodata jobs list --me
macrodata jobs list --json
macrodata jobs list --cursor <next_cursor>
```

### `macrodata jobs get`

Fetches the main job summary. This includes status, progress, starter identity, stage layout, worker counts, and per-stage step structure. Stage rows include shard completion and worker `running/completed/total` counts.

Options:

- `--json`

```bash
macrodata jobs get <job_id>
```

### `macrodata jobs attach`

Reattaches the cloud job console for an existing cloud job. The attached view shows the current job summary header, a capped live worker log view, and exits automatically when the remote job reaches a terminal state.

```bash
macrodata jobs attach <job_id>
```

### `macrodata jobs manifest`

Reads the captured run manifest for a job. Text mode always prints the runtime section first. Use the optional flags to additionally show dependency information and captured code metadata.

Options:

- `--show-deps`
- `--show-code`
- `--json`

```bash
macrodata jobs manifest <job_id>
macrodata jobs manifest <job_id> --show-deps
macrodata jobs manifest <job_id> --show-code
```

### `macrodata jobs workers`

Lists workers for a job, optionally scoped to one stage. The response includes worker UUIDs, names, status, host, start/end times, and shard counters.

Options:

- `--stage <stage_index>`
- `--limit <count>`
- `--cursor <cursor>`
- `--json`

```bash
macrodata jobs workers <job_id> --stage 0
macrodata jobs workers <job_id> --limit 50
macrodata jobs workers <job_id> --json
macrodata jobs workers <job_id> --cursor <next_cursor>
```

### `macrodata jobs logs`

Fetches cloud worker or service logs for a job. You can narrow by stage, worker, source, severity, search text, and time window.
Without `--search`, the CLI defaults to the most recent hour if you do not pass a window.
With `--search`, the window must be explicit and the stage is mandatory so the query shape stays transparent and bounded.

Options:

- `--stage <stage_index>`
- `--worker <worker_id>`
- `--source-type worker|service`
- `--source-name <name>`
- `--severity info|warning|error`
- `--search <text>`
- `--start-ms <epoch_ms>`
- `--end-ms <epoch_ms>`
- `--cursor <cursor>`
- `--limit <count>`
- `--follow`
- `--json`

Notes:

- `--search` requires `--stage`
- `--search` requires both `--start-ms` and `--end-ms`
- `--search` supports at most 100 results per request
- `--cursor` reuses `nextCursor` from a previous response to fetch the next page in the same window
- one-shot log fetches default to `100` entries per request
- `--follow` defaults to `500` entries per request
- `--follow` may skip older backlog under sustained log volume to stay live; when it does, it prints the skipped timestamp range and recovery guidance
- `--follow` cannot be combined with `--json`, `--cursor`, or `--search`

```bash
macrodata jobs logs <job_id>
macrodata jobs logs <job_id> --json
macrodata jobs logs <job_id> --start-ms 1713340800000 --end-ms 1713341700000 --cursor <next_cursor>
macrodata jobs logs <job_id> --follow
macrodata jobs logs <job_id> --stage 0 --severity error
macrodata jobs logs <job_id> --stage 0 --search retry --start-ms 1713340800000 --end-ms 1713341700000 --limit 50
```

### `macrodata jobs metrics`

Fetches cloud step metrics for one stage.
The default response is inventory only: each step and the metric labels available under it.
Add `--step` to inspect one step, and add one or more `--metric` labels to fetch actual values for that step.

Options:

- `--step <step_index>`
- `--metric <label>`; may be repeated and requires `--step`
- `--json`

```bash
macrodata jobs metrics <job_id> <stage_index>
macrodata jobs metrics <job_id> <stage_index> --step 2
macrodata jobs metrics <job_id> <stage_index> --step 2 --metric rows_processed --metric queue_depth
```

### `macrodata jobs resource-metrics`

Fetches cloud resource telemetry for one stage. This includes time-bucketed CPU, memory, and network usage, plus resource-event activity over the selected time window.

Options:

- `--range 5m|15m|1h|4h|6h|24h|7d`
- `--worker-id <worker_id>`; may be repeated
- `--start-ms <epoch_ms>`
- `--end-ms <epoch_ms>`
- `--bucket-count <count>`
- `--json`

```bash
macrodata jobs resource-metrics <job_id> <stage_index>
macrodata jobs resource-metrics <job_id> <stage_index> --worker-id worker-1
```

### `macrodata jobs cancel`

Requests cancellation for a pending or running cloud job and returns the control-plane cancellation result.

Options:

- `--json`

```bash
macrodata jobs cancel <job_id>
```

Every job-inspection command supports `--json`. That mode prints the raw API response so scripts and agentic tools can consume the exact payload returned by Macrodata.

## Notes

- `launch_local(...)` does not require Macrodata auth
- `launch_cloud(...)` requires Macrodata auth
- `macrodata jobs logs ...`, `macrodata jobs metrics ...`, and `macrodata jobs resource-metrics ...` are only available for cloud jobs
- `macrodata jobs resource-metrics ...` accepts at most 50 distinct `--worker-id` filters per request
- `macrodata jobs cancel ...` is only available for cloud jobs in `pending` or `running` state

## Internal Notes

- The CLI forwards job-inspection reads and cancellation to the Macrodata control plane under `/api/cli/jobs/...`
- `--json` is intended as the stable machine-readable contract; the default text output is best-effort formatting on top of that raw response
