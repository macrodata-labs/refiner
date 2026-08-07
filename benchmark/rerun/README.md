# Rerun Benchmarks

This folder contains cloud benchmark harnesses for Rerun RRD workloads.

- `run_cloud_benchmark.py`: submits Macrodata Cloud jobs for Rerun read,
  robotics conversion, and Rerun write paths, then writes JSON artifacts with
  job ids, stage timings, metrics, and output inspection.
- `compare_results.py`: compares two benchmark `summary.json` artifacts and
  prints case-level and stage-level timing deltas.
- `refresh_aws_secrets.py`: copies short-lived credentials from an AWS CLI
  profile into the Macrodata workspace secret environment used by cloud jobs.
- `run_local_benchmark.py`: runs a local single-recording RRD copy benchmark
  that compares the direct byte-copy path with the chunk-selection fallback.
- `run_cleanup_benchmark.py`: runs a local benchmark for the default-root RRD
  cleanup matcher used by `FileCleanupReducerSink`.

The default inputs are the ten base RRD files from:

```text
s3://macrodata-rerun-format-tests/dominique-sample/
```

The default cases are:

- `recording-summary`: `read_rerun(output="recording")`, summarize timeline
  and static tables, write JSONL.
- `robotics-summary`: `read_rerun(output="robotics")` for action/state paths,
  summarize frame rows and vector widths, write JSONL.
- `rrd-copy`: `read_rerun(output="recording", materialize_tables=False)`
  followed by `write_rerun(...)` to exercise the distributed RRD writer's raw
  chunk path without timing unused Arrow table materialization.

These cases intentionally cover both the high-fidelity recording path and the
robotics convenience path. Do not remove a case just to make a performance run
look better.

## Prerequisites

- The current branch must be pushed and available on a GitHub PR before cloud
  launch.
- Macrodata CLI auth must be configured.
- Workspace secrets in the selected environment must include AWS credentials
  for the source/output S3 bucket. The default environment is `researcher`.
  Pass `--aws-profile` explicitly when refreshing those credentials; the helper
  intentionally does not fall back to the AWS default profile.

If local AWS credentials are valid, refresh the cloud secret environment without
printing credential values:

```bash
uv run python benchmark/rerun/refresh_aws_secrets.py \
  --aws-profile 210049840512_Researcher \
  --secret-env researcher
```

## Run

```bash
REFINER_ATTACH=detach uv run python benchmark/rerun/run_cloud_benchmark.py
```

Useful options:

- `--case robotics-summary --case rrd-copy` to run a subset.
- `--iterations 3` to repeat each case.
- `--input s3://bucket/path/file.rrd` to use custom inputs; repeat as needed.
- `--output-root s3://bucket/prefix` to choose where cloud outputs are written.
- `--num-workers 4` to vary cloud parallelism.
- `--aws-profile 210049840512_Researcher` to inspect S3 outputs locally with a
  specific profile after cloud completion.
- `--continue-on-failure` to keep launching later cases after one case fails.
  By default the harness records the failed case and stops, so bad credentials or
  setup failures do not create a misleading benchmark session.

For a local smoke benchmark that does not require cloud credentials:

```bash
uv run python benchmark/rerun/run_local_benchmark.py
```

For the reducer cleanup matcher benchmark:

```bash
uv run python benchmark/rerun/run_cleanup_benchmark.py
```

The local benchmark generates a synthetic single-recording RRD, then measures
the direct-copy branch against the chunk-selection fallback on the same source
file. Use `--writes-per-iteration` to repeat the same shard write within one
timed run when you want to amplify per-row writer overhead.

For cloud runs, the summary also records `stage_duration_s`, the sum of stage
durations. That is often a better performance signal than wall time because it
excludes queueing noise from the cloud scheduler.

Artifacts are written under `benchmark/rerun/artifacts/` by default:

- one per-case result JSON
- one summary JSON for the benchmark session

Each case records `planned_shards`. RRD files are file-atomic, so runs where
`planned_shards < --num-workers` can underutilize workers and should not be used
as scaling evidence.

## Compare

After running a baseline and candidate benchmark, compare their summaries:

```bash
uv run python benchmark/rerun/compare_results.py \
  benchmark/rerun/artifacts/baseline/summary.json \
  benchmark/rerun/artifacts/candidate/summary.json
```

Only completed jobs are used for timing deltas. Failed jobs still appear in the
run-count columns so setup problems are visible instead of silently averaged in.
Planned shard counts and shard-planning warnings are printed with the timing
table.
