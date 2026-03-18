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
until no work remains. In-process debugging still respects the same reader and
planning model, but runs inside one Python process.

Use reader-specific options such as `target_shard_bytes` and `num_shards` to
change shard granularity.

## What Python Functions Receive

Refiner has two execution styles:

- Python UDF steps such as `map(...)`, `flat_map(...)`, `filter(...)`, and `batch_map(...)`
- expression-backed vectorized steps such as `select(...)`, `with_columns(...)`, and `filter(expr)`

The practical Python contract is:

- `map(...)` receives one `Row`
- `map_async(...)` receives one `Row`
- `flat_map(...)` receives one `Row`
- `filter(...)` with a Python predicate receives one `Row`
- `batch_map(...)` receives `list[Row]`

Even though the engine may use Arrow-backed `Tabular` blocks internally between
segments, Python UDFs do not receive raw Arrow tables directly. If you want
Arrow-backed execution, use the expression API instead of a Python UDF.

### `Row` assumptions

A `Row` is:

- mapping-like (`row["column"]`)
- immutable from the caller's perspective
- either dict-backed or a lightweight view over tabular data

Safe assumptions:

- `row["col"]` works
- `dict(row.items())` works
- `row.update(...)` / `row.drop(...)` / `row.pop(...)` return modified rows

Unsafe assumptions:

- every row is a plain `dict`
- mutating a row in place is part of the API
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
LeRobot-specific helpers/views.

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

- in-process debugging helpers such as `iter_rows()`, `take()`, and `materialize()` do not write output
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

### Performance model

The writer is optimized around the natural work unit:

- one pipeline row = one episode

Within a shard:

- frame parquet writing is incremental and synchronous
- video work is scheduled per episode and prepared asynchronously
- shard-local metadata is buffered and flushed once the shard completes

This split matters because frame parquet writes are Arrow-friendly and cheap to
append, while video work is dominated by file IO, source probing, remux or
transcode decisions, and codec work.

### `max_video_prepare_in_flight`

`max_video_prepare_in_flight` controls how many episode-level video preparation
tasks can be in flight at once inside one worker.

Why it helps:

- overlapping remote opens and source probing hides latency
- several episodes can prepare videos while frame parquet writes keep moving
- it prevents unbounded fan-out when a shard contains many video-bearing episodes

Why it is bounded:

- video preparation is heavy on file handles, network IO, and codec resources
- too much concurrency just thrashes the worker instead of increasing throughput

### Remux vs transcode

The writer prefers remux when it can, and falls back to transcode when it must.

Remux is used when:

- the source clip and requested output are container/codec-compatible
- the requested time span can be represented safely without decoding/re-encoding
- existing source video stats are allowed to be reused

Transcode is used when:

- the requested output codec or pixel format requires it
- the clip boundaries are not remux-safe
- `force_recompute_video_stats=True`
- source video stats are missing or were invalidated by an upstream transform

Why remux is faster:

- it avoids decoding frames
- it avoids re-encoding frames
- it mostly moves packets into the output container

Why transcode is sometimes required:

- recomputing video stats needs decoded frames
- incompatible source/output encoding details need re-encoding
- exact clip extraction is not always packet-safe

### Threading assumptions

`transencoding_threads` is treated as a per-worker budget for transcode work.
When several video streams exist on the same row, the writer divides that
budget across them so each stream writer does not try to claim the whole worker.

This matters because separate video writers can each spawn codec threads.
Without that division, multi-camera episodes oversubscribe CPUs quickly.

Stage 2 reduction writes final:

- `meta/episodes/chunk-000/file-000.parquet`
- `meta/tasks.parquet`
- `meta/info.json`
- `meta/stats.json`

while cleaning up stage-1 `meta/chunk-*` metadata and any non-finalized
`data/chunk-*` / `videos/.../chunk-*` payloads.

## Related Pages

- [Pipeline basics](pipeline-basics.md)
- [Expression transforms](expression-transforms.md)
- [In-process debugging](in-process-debugging.md)
- [Launchers](launchers.md)
- [Robotics](robotics.md)
