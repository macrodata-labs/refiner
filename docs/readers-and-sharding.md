---
title: "Readers and Sharding"
description: "Choose input readers and understand shard planning behavior"
---

Refiner includes built-in readers for tabular and episode-oriented datasets.

## Available Readers

- `read_csv(...)`
- `read_jsonl(...)`
- `read_parquet(...)`
- `read_lerobot(...)`

Each returns a `RefinerPipeline` with that reader as the source.

## Shard Model

Readers expose shards as units of work. A shard is identified by `path`, `start`, and `end`.

- CSV/JSONL: shard ranges are byte-oriented.
- Parquet: shard ranges are row-group-based (`rowgroups`) or planned by bytes (`bytes_lazy`).
- LeRobot: shard ranges are per-file over `meta/episodes/*.parquet` (one shard per episode parquet file).

## Reader Notes for Users

- CSV and JSONL readers yield dict-backed rows.
- Parquet reader yields row views that are converted on access.
- LeRobot reader emits one row per episode with:
  - `frames`: list of frame dicts for the episode slice.
  - video feature columns from `meta/info.json.features` where `dtype == "video"`, emitted as `VideoFile` handles.
  - `metadata`: dict with shared `lerobot_info`, shared dataset-level `lerobot_stats`, shared dataset-level `lerobot_tasks` as a `{task_index: task}` mapping loaded from `meta/tasks.parquet`, and per-episode `lerobot_episode_stats`.
  - episode-level `tasks` from LeRobot metadata are omitted from emitted rows so edits go through the canonical task table plus frame `task_index`.
  - raw transport metadata keys under `stats/*`, `videos/*`, and `meta/episodes/*` are omitted from emitted rows.
  - frame slicing requires `dataset_from_index`/`dataset_to_index` in each episode row.
  - optional `limit` bounds emitted episodes per reader instance.
  - `media_max_in_flight` controls concurrent episode materialization work inside the reader.
  - `media_preserve_order` controls whether those async reader results are yielded in input order.
  - `read_lerobot(...)` accepts either one dataset root or a list of dataset roots on the same filesystem/protocol.
  - when reading multiple dataset roots, the reader merges all source `meta/tasks.parquet` tables once in input order, keeps the first-seen task ids stable, appends unseen tasks at the end, and remaps per-frame `task_index` values into that merged task table before emitting rows.
- Use `target_shard_bytes` to control shard granularity.

## LeRobot Writer Tuning

`write_lerobot(...)` keeps video tuning grouped:

- `video=mdr.LeRobotVideoConfig(...)` groups codec, pixel format, encoder threads, decoder threads, and encoder options.
- `stats=mdr.LeRobotStatsConfig(...)` groups clip sampling stride and quantile bin count.
- Task metadata is canonical: `write_lerobot(...)` resolves frame `task_index` values through `metadata.lerobot_tasks`, derives episode-level `tasks`, treats top-level `task` as an ordinary passthrough column rather than canonical task metadata, and raises if any frame task index cannot be mapped.
- `write_lerobot(...)` writes `meta/tasks.parquet` with plain `task` and `task_index` columns. The reader still accepts legacy LeRobot parquet files that store task names under `__index_level_0__`.
- Video stats are always written; use a larger `stats.sample_stride` when you want cheaper sampling.
- When the sink receives consecutive LeRobot episodes that span an entire source video file, it can remux that file into the output without decoding frames.
- When consecutive episodes span several whole compatible source files, the sink can remux those files into one output file without decoding frames.
- When an episode clip starts on a source keyframe and ends on an exact packet boundary, the sink can remux that aligned subsegment without transcoding.
- Remux output is written directly as fragmented MP4 to the destination stream; it does not stage a separate local temp file first.
- Video entries in `meta/info.json.features` are written in LeRobot-style channel-first form with an `info` block for fps, size, channels, codec, and pixel format.

## Internal Notes

- Parquet byte-lazy mode maps planned byte ranges to row groups at read time.
- Parquet row access uses batch-level cached column-name indexing for faster key lookup.
- LeRobot expects parquet metadata under `meta/episodes/**`; legacy JSONL metadata is not used.
- LeRobot reads `fps`, `robot_type`, `features`, `data_path`, and `video_path` from `meta/info.json` when present.
- The LeRobot writer is batch-oriented per shard block: frame parquet writes happen per batch table, each `video_key` uses one batch-scoped `VideoWriter.write_videos(...)` call that prepares videos concurrently and commits them in order, and episode rows are finalized from an async queue of completed video results.
