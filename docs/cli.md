---
title: "CLI"
description: "Macrodata CLI commands used with Refiner"
---

Refiner installs the `macrodata` CLI.

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

- `--logs all|none|one|errors`
- script arguments after `--`

Behavior:

- `Ctrl+C` exits with code `130`
- local launcher resume/failure messages are printed cleanly, while ordinary script exceptions still surface normally
- the script directory is added to `sys.path`, so sibling imports work the same way they do with `python script.py`

```bash
macrodata run path/to/pipeline.py
```

```bash
macrodata run --logs one path/to/pipeline.py -- --workers 4 --rows 20
```

## Credential Lookup

See [Auth](auth.md) for the shared credential lookup order and credential file location.

## Job Inspection

Inspect jobs in the workspace attached to your current API key:

### `macrodata jobs list`

Options:

- `--status <status>`
- `--kind local|cloud`
- `--limit <count>`
- `--me`
- `--cursor <opaque_cursor>`
- `--json`

```bash
macrodata jobs list --kind cloud
macrodata jobs list --me
```

### `macrodata jobs get`

Options:

- `--json`

```bash
macrodata jobs get <job_id>
```

### `macrodata jobs manifest`

Options:

- `--show-runtime`
- `--show-deps`
- `--show-code`
- `--json`

```bash
macrodata jobs manifest <job_id>
macrodata jobs manifest <job_id> --show-runtime --show-deps
```

### `macrodata jobs workers`

Options:

- `--stage <stage_index>`
- `--limit <count>`
- `--cursor <opaque_cursor>`
- `--json`

```bash
macrodata jobs workers <job_id> --stage 0
macrodata jobs workers <job_id> --limit 50
macrodata jobs workers <job_id> --cursor <opaque_cursor>
```

### `macrodata jobs logs`

Options:

- `--stage <stage_index>`
- `--worker <worker_id>`
- `--source-type worker|service`
- `--source-name <name>`
- `--severity info|warning|error`
- `--search <text>`
- `--start-ms <epoch_ms>`
- `--end-ms <epoch_ms>`
- `--limit <count>`
- `--json`

```bash
macrodata jobs logs <job_id>
macrodata jobs logs <job_id> --stage 0 --severity error
```

### `macrodata jobs metrics`

Options:

- `--step <step_index>`
- `--json`

```bash
macrodata jobs metrics <job_id> <stage_index>
macrodata jobs metrics <job_id> <stage_index> --step 2
```

### `macrodata jobs resource-metrics`

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
