# Rerun Benchmarks

This folder contains cloud benchmark harnesses for Rerun RRD workloads.

- `run_cloud_benchmark.py`: submits Macrodata Cloud jobs for Rerun read,
  robotics conversion, and Rerun write paths, then writes JSON artifacts with
  job ids, stage timings, metrics, and output inspection.

The default inputs are the ten base RRD files from:

```text
s3://macrodata-rerun-format-tests/dominique-sample/
```

The default cases are:

- `recording-summary`: `read_rerun(output="recording")`, summarize timeline
  and static tables, write JSONL.
- `robotics-summary`: `read_rerun(output="robotics")` for action/state paths,
  summarize frame rows and vector widths, write JSONL.
- `rrd-copy`: `read_rerun(output="recording").write_rerun(...)` to exercise the
  distributed RRD writer's raw chunk path.

These cases intentionally cover both the high-fidelity recording path and the
robotics convenience path. Do not remove a case just to make a performance run
look better.

## Prerequisites

- The current branch must be pushed and available on a GitHub PR before cloud
  launch.
- Macrodata CLI auth must be configured.
- Workspace secrets in the selected environment must include AWS credentials
  for the source/output S3 bucket. The default environment is `researcher`.

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

Artifacts are written under `benchmark/rerun/artifacts/` by default:

- one per-case result JSON
- one summary JSON for the benchmark session
