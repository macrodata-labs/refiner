---
title: "Readers And Sharding"
description: "How Refiner readers emit data, plan shards, and feed Python code"
---

Readers decide two things:

- what data shape enters the pipeline
- how that input is split into shards for launched execution

## Built-In Readers

- `read_csv(...)`
- `read_jsonl(...)`
- `read_parquet(...)`
- `read_lerobot(...)`

Each reader returns a `RefinerPipeline`.

## Shard Model

A shard is the unit of work claimed by a worker.

Common shard metadata includes:

- `shard_id`
- `path`
- `start`
- `end`

For launched execution, workers claim shards from the runtime lifecycle backend
until no work remains. Local iteration still respects the same reader/planning
model, but runs in-process.

Use reader-specific options such as `target_shard_bytes` and `num_shards` to
change shard granularity.

## What Python Functions Receive

Refiner has two execution styles:

- Python UDF steps such as `map(...)`, `flat_map(...)`, `filter(...)`, and `batch_map(...)`
- expression-backed vectorized steps such as `select(...)`, `with_columns(...)`, and `filter(expr)`

The important contract for Python code is:

- `map(...)` receives one `Row`
- `map_async(...)` receives one `Row`
- `flat_map(...)` receives one `Row`
- `filter(...)` with a Python predicate receives one `Row`
- `batch_map(...)` receives `list[Row]`

Even though the engine may use Arrow-backed `Tabular` blocks internally between
segments, Python UDFs do not receive raw Arrow tables directly. If you want
Arrow-backed/vectorized behavior, use the expression API instead of a Python UDF.

### `Row` assumptions

A `Row` is:

- mapping-like (`row["column"]`)
- immutable from the caller's perspective
- either dict-backed or a lightweight view over tabular data

Use:

- `row.update(...)` to return a patched row
- `row.drop(...)` to hide keys
- `row.pop(...)` for persistent pop semantics

Do not assume:

- the row is a plain `dict`
- mutating nested values in place is part of the execution contract
- every reader yields the same concrete row subclass

## Reader Behavior

### CSV and JSONL

- rows are dict-backed
- shard planning is byte-oriented

### Parquet

- rows are lightweight views over Arrow-backed tabular data
- shard planning is row-group-based, or lazily refined from byte targets
- row values are converted to Python on access

### LeRobot

`read_lerobot(...)` emits one episode row at a time.

Each emitted row is a `LeRobotRow`, which wraps a normal base row and adds
LeRobot-specific views/helpers.

A LeRobot episode row includes:

- ordinary episode-level columns from `meta/episodes`
- `frames`
  - usually a `LeRobotTabular` / `Tabular` containing that episode's frame rows
- `metadata`
  - a `LeRobotMetadata` dataclass carrying dataset-level `info`, `stats`, and canonical tasks
- video feature columns
  - exposed as `VideoFile` handles through the LeRobot row helpers

The reader also applies a few format assumptions:

- one pipeline row represents one episode
- frame slicing is driven by `dataset_from_index` / `dataset_to_index`
- the canonical task table is loaded from `meta/tasks.parquet`
- when several dataset roots are read together, task ids are merged once in input order and per-frame `task_index` values are remapped to that merged table
- transport-only keys such as `videos/*` placement fields remain available in the base row, but the high-level API is meant to go through `LeRobotRow.videos`, `LeRobotRow.stats`, `LeRobotRow.metadata`, and `LeRobotRow.frames`

## LeRobot Writer Notes

`write_lerobot(...)` is a deferred sink:

- local iteration helpers such as `iter_rows()`, `take()`, and `materialize()` do not write output
- actual LeRobot files are produced in launched execution (`launch_local(...)`, `launch_cloud(...)`)

Current writer API:

- `output`
- `data_files_size_in_mb`
- `video_files_size_in_mb`
- `max_video_prepare_in_flight`
- `codec`
- `pix_fmt`
- `transencoding_threads`
- `encoder_options`
- `quantile_bins`
- `force_recompute_video_stats`

Important writer assumptions:

- input rows are expected to be LeRobot-shaped episode rows
- `metadata` must be present on every row as `LeRobotMetadata`
- per-frame `task_index` is canonical; episode-level `tasks` are rebuilt from the frame table
- `fps` and `robot_type` must stay stable within a written shard
- frame parquet data is written incrementally
- episode metadata rows are buffered per shard and flushed at shard completion
- video stats are dropped and recomputed when upstream transforms invalidate them

Stage 2 reduction writes final:

- `meta/episodes/chunk-000/file-000.parquet`
- `meta/tasks.parquet`
- `meta/info.json`
- `meta/stats.json`

while cleaning up stage-1 `meta/chunk-*` metadata and any non-finalized
`data/chunk-*` / `videos/.../chunk-*` payloads.

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Local execution](local-execution.md)
- [Expression transforms](expression-transforms.md)
- [Launchers](launchers.md)
- [Worker runtime](worker-runtime.md)
