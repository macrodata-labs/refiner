---
title: "CLI"
description: "Macrodata CLI commands used with Refiner"
---

Refiner installs the `macrodata` CLI.

## Auth

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

### Login

```bash
macrodata login
```

Non-interactive options:

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

```bash
macrodata run path/to/pipeline.py
```

Optional flags:

```bash
macrodata run --logs one path/to/pipeline.py -- --workers 4 --rows 20
```

`macrodata run` currently supports:

- `--logs all|none|one|errors`
- `Ctrl+C` exits with code `130`
- local launcher resume/failure messages are printed cleanly, while ordinary script exceptions still surface normally
- the script directory is added to `sys.path`, so sibling imports work the same way they do with `python script.py`

## Credential Lookup

See [Auth](auth.md) for the shared credential lookup order and credential file location.

## Job Inspection

Inspect jobs in the workspace attached to your current API key:

```bash
macrodata jobs list --kind cloud
macrodata jobs get <job_id>
```

Read the captured run manifest:

```bash
macrodata jobs manifest <job_id>
```

Inspect workers for a specific stage:

```bash
macrodata jobs workers <job_id> --stage 0
```

Fetch cloud-job logs from the last hour:

```bash
macrodata jobs logs <job_id>
macrodata jobs logs <job_id> --stage 0 --severity error
```

Fetch cloud-job metrics:

```bash
macrodata jobs metrics <job_id>
macrodata jobs metrics <job_id> --stage 0 --range 6h
```

Every job-inspection command supports `--json`. That mode prints the raw API response so scripts and agentic tools can consume the exact payload returned by Macrodata.

## Notes

- `launch_local(...)` does not require Macrodata auth
- `launch_cloud(...)` requires Macrodata auth
- `macrodata jobs logs ...` and `macrodata jobs metrics ...` are only available for cloud jobs

## Internal Notes

- The CLI forwards job-inspection reads to the Macrodata control plane under `/api/cli/jobs/...`
- `--json` is intended as the stable machine-readable contract; the default text output is best-effort formatting on top of that raw response
