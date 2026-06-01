---
title: "Jobs, Logs, and Metrics"
description: "Inspect cloud jobs from the Macrodata CLI"
---

# Jobs, Logs, And Metrics

Job inspection commands default to terminal-friendly output. Add `--json` when
an agent, script, or CI job needs structured data.

## List Jobs

```bash
macrodata jobs list
```

`jobs list` prints recent jobs in the current workspace. Use it to find a job
ID, check whether a cloud run is still active, or page through older runs.

Useful filters:

```bash
macrodata jobs list --me --kind cloud --status running --limit 50
macrodata jobs list --cursor <next_cursor> --json
```

Example output:

```text
ID       Name                 Status    Kind   Created                  Started By
job_123  aloha preprocessing  running   cloud  2026-05-28 12:40:10 UTC ada@example.com
job_122  local smoke test     done      local  2026-05-28 11:58:31 UTC ada@example.com

Next cursor: macrodata jobs list --limit 20 --cursor eyJwYWdlIjoyfQ
```

| Option | Use |
| --- | --- |
| `--status <status>` | Filter by job status |
| `--kind local\|cloud` | Filter by executor kind |
| `--limit <n>` | Maximum jobs to return |
| `--me` | Only jobs started by the authenticated user |
| `--cursor <cursor>` | Continue a paginated listing |
| `--json` | Print raw JSON |

## Inspect One Job

```bash
macrodata jobs get job_123
macrodata jobs get job_123 --json
```

`get` prints job status, tracking URL, cost, timestamps, stage progress,
runtime resources, step summaries, and availability of manifest, logs, and
metrics.

With `--json`, `get` prints the normalized job object directly.

Example output:

```text
Job: aloha preprocessing  ID: job_123  URL: https://app.macrodata.ai/jobs/job_123
Status: running  Kind: cloud  Cost: $1.25
Created: 2026-05-28 12:40:10 UTC  Started: 2026-05-28 12:40:22 UTC
Created By: ada@example.com
Available: manifest, logs, metrics

Stages
Index  Name    Status   Progress
0      ingest  running  run=2 done=1 tot=4

Steps
Stage  Step  Name          Type  Columns
0      2     enrich_rows   map   columns=18
```

For local jobs, the summary also prints the run directory.

## Attach To A Running Job

```bash
macrodata jobs attach job_123
```

`attach` reconnects to a running cloud job and follows the same terminal view
used by attached launches.

It is only valid for cloud jobs. If the job has already finished, use
`jobs get`, `jobs logs`, and `jobs metrics` instead.

Example output starts with the live job header, then updates progress and log
lines until the job reaches a terminal state or you interrupt it:

```text
Job: aloha preprocessing  ID: job_123  Status: running
Stage 0 ingest: run=2 done=1 tot=4
2026-05-28T12:42:11Z | INFO | worker=worker_1 | decoded shard 17
```

## Manifest

```bash
macrodata jobs manifest job_123
macrodata jobs manifest job_123 --deps
macrodata jobs manifest job_123 --code
macrodata jobs manifest job_123 --json
```

`manifest` prints the runtime environment captured when the job was submitted:
Python version, platform, package dependencies, script path, script checksum,
and optionally the captured script text. This is the command to use when you
need to reproduce or audit exactly what ran.

Example output:

```text
Runtime
Python: 3.12.3
Platform: linux-x86_64

Dependencies: 27 dependencies (rerun with --deps)

Code
Path: pipeline.py
SHA256: 4c21...
Source: (rerun with --code)
```

| Option | Use |
| --- | --- |
| `--deps` | Show dependencies captured in the manifest |
| `--code` | Show captured script text |
| `--json` | Print raw JSON |

Use `manifest --json` for exact machine-readable environment, dependency, and
script metadata.

## Workers

```bash
macrodata jobs workers job_123
macrodata jobs workers job_123 --stage 0 --limit 50
macrodata jobs workers job_123 --cursor <next_cursor> --json
```

`workers` lists the worker processes that executed a job. Use it to identify
slow or failed workers, inspect shard counts, and get worker IDs for log and
resource-metric filters.

Example output:

```text
ID        Name      Stage  Status   Shards Running  Shards Done  Started                  Ended
worker_1  worker-a  0      running  1               12           2026-05-28 12:41:03 UTC  -
worker_2  worker-b  0      done     0               14           2026-05-28 12:41:03 UTC  2026-05-28 12:45:12 UTC

Next cursor: macrodata jobs workers job_123 --stage 0 --limit 20 --cursor eyJwYWdlIjoyfQ
```

| Option | Use |
| --- | --- |
| `--stage <index>` | Filter by stage index |
| `--limit <n>` | Maximum workers to return |
| `--cursor <cursor>` | Continue a paginated worker listing |
| `--json` | Print raw JSON |

## Follow Logs

```bash
macrodata jobs logs job_123 --follow
```

`logs` fetches stdout, stderr, and service logs for a cloud job. Use one-shot
mode for historical windows and `--follow` when you want a live stream.

Logs can be filtered before printing or streaming:

```bash
macrodata jobs logs job_123 --stage 0 --worker worker_123
macrodata jobs logs job_123 --severity error --search traceback
macrodata jobs logs job_123 --source-type service --source-name vllm
macrodata jobs logs job_123 --start-ms 1760000000000 --end-ms 1760000300000
macrodata jobs logs job_123 --limit 500 --json
```

Example output:

```text
2026-05-28T12:42:11.120Z | INFO    | stage=0 worker=worker_1 | decoded shard 17
2026-05-28T12:42:15.502Z | WARNING | stage=0 worker=worker_1 | retrying shard 18

More logs are available after these results.
Next cursor: macrodata jobs logs job_123 --stage 0 --limit 500 --cursor eyJwYWdlIjoyfQ
```

`--follow` cannot be combined with `--json`, `--cursor`, or `--search`. A
search requires an explicit `--stage` and a time window with `--start-ms` and
`--end-ms`.

| Option | Use |
| --- | --- |
| `--stage <index>` | Filter by stage index |
| `--worker <id>` | Filter by worker ID |
| `--source-type worker\|service` | Filter by source type |
| `--source-name <name>` | Filter by source name |
| `--severity info\|warning\|error` | Filter by severity |
| `--search <text>` | Case-insensitive substring filter |
| `--start-ms <epoch_ms>` | Window start time |
| `--end-ms <epoch_ms>` | Window end time |
| `--cursor <cursor>` | Continue a paginated log listing |
| `--limit <n>` | Maximum log entries |
| `--follow` | Poll continuously for new entries |
| `--json` | Print raw JSON |

## Stage Metrics

```bash
macrodata jobs metrics job_123 0
```

`metrics` reads user metrics emitted by Refiner steps in a stage. Without a
step, it prints metric inventory so you can discover labels before fetching
values.

Without `--step`, this lists metric inventory for the stage. Fetch values by
selecting a step and one or more metric labels:

```bash
macrodata jobs metrics job_123 0 --step 2 --metric rows_processed
macrodata jobs metrics job_123 0 --step 2 --metric rows_processed --metric decode_ms
macrodata jobs metrics job_123 0 --step 2 --metric rows_processed --workers --desc
macrodata jobs metrics job_123 0 --step 2 --metric rows_processed --worker worker_123 --json
```

Inventory output:

```text
Job: job_123  Stage: 0
Detail: inventory

Step 2: enrich_records (map)
Metric          Kind
rows_processed counter
decode_ms      timer

rerun with --step <index> --metric <label> to fetch metric values
```

Value output:

```text
Job: job_123  Stage: 0
Detail: values

Step 2: enrich_records (map)
rows_processed (counter)
Total: 120000
Rate (lifetime): 231.2/s
Per Worker (lifetime): 30000
```

| Option | Use |
| --- | --- |
| `--step <index>` | Fetch values for one step |
| `--metric <label>` | Metric label to fetch; repeatable |
| `--workers` | Include worker rankings for supported metrics |
| `--worker <id>` | Filter values/rankings to worker IDs; repeatable |
| `--asc` | Sort worker rankings ascending |
| `--desc` | Sort worker rankings descending |
| `--json` | Print raw JSON |

`--metric`, `--worker`, and `--workers` require `--step`. `--worker` also
requires `--metric`. `--asc` and `--desc` apply to worker rankings.

## Resource Metrics

```bash
macrodata jobs resource-metrics job_123 0
```

`resource-metrics` reads runtime resource samples for a stage: CPU, memory,
network, and accelerator usage where available. Use it to debug resource
limits, GPU utilization, and worker imbalance.

Resource metrics cover CPU, memory, network, and runtime resource samples:

```bash
macrodata jobs resource-metrics job_123 0 --range 15m
macrodata jobs resource-metrics job_123 0 --worker-id worker_123 --worker-id worker_456
macrodata jobs resource-metrics job_123 0 --start-ms 1760000000000 --end-ms 1760000300000 --bucket-count 120 --json
```

Example output:

```text
Job: job_123  Stage: 0
Range: 15m
Latest sample: 2026-05-28 12:45:00 UTC
CPU: 3100m / 4000m  Memory: 12.4 GiB / 16 GiB
Network In: 183 MB  Network Out: 42 MB
Samples: 240
```

| Option | Use |
| --- | --- |
| `--range 5m\|15m\|1h\|4h\|6h\|24h\|7d` | Metrics range |
| `--worker-id <id>` | Filter by worker ID; repeatable |
| `--start-ms <epoch_ms>` | Window start time |
| `--end-ms <epoch_ms>` | Window end time |
| `--bucket-count <n>` | Requested number of buckets |
| `--json` | Print raw JSON |

## Cancel

```bash
macrodata jobs cancel job_123
macrodata jobs cancel job_123 --json
```

`cancel` requests cancellation for a cloud job. It is idempotent from the
operator's perspective: use `jobs get` afterward to confirm the terminal state.

Example output:

```text
Canceled: job_123
  Requested: 4
  Canceled: 4
  Failed: 0
```

## Related Pages

- [Observability](../running-pipelines/observability.md)
- [Cloud Jobs and Files](../platform/cloud-jobs-and-files.md)
