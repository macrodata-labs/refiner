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
  - video feature columns from `meta/info.json.features` where `dtype == "video"`, emitted as `Video` handles.
  - `metadata`: dict with `lerobot_info`, dataset-level `lerobot_stats`, and per-episode `lerobot_episode_stats`.
  - raw transport metadata keys under `stats/*`, `videos/*`, and `meta/episodes/*` are omitted from emitted rows.
  - frame slicing requires `dataset_from_index`/`dataset_to_index` in each episode row.
  - optional `limit` bounds emitted episodes per reader instance.
  - `media_max_in_flight` controls concurrent episode materialization work inside the reader.
  - `media_preserve_order` controls whether those async reader results are yielded in input order.
  - `read_lerobot(...)` accepts either one dataset root or a list of dataset roots on the same filesystem/protocol.
- Use `target_shard_bytes` to control shard granularity.

## Hydrating External Files

Use `refiner.hydrate_media(...)` with `.map_async(...)` when you want decoded
video clips materialized in-process.

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("s3://bucket/dataset")
    .map_async(
        mdr.hydrate_media("observation.images.main"),
        max_in_flight=8,
    )
)
```

`hydrate_media(...)` currently accepts `Video` values only. For LeRobot inputs,
that means clip-aligned hydration is explicit and decode-backed:

- use `.map_async(...)` to control concurrency
- expect decoded frames in `video.media`

Rows are yielded in input order unless you explicitly choose otherwise on the async step.

## LeRobot Writer Tuning

`write_lerobot(...)` keeps video tuning grouped:

- `video=mdr.LeRobotVideoConfig(...)` groups codec, pixel format, encoder threads, decoder threads, and encoder options.
- `stats=mdr.LeRobotStatsConfig(...)` groups clip sampling stride and quantile bin count.
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
- The LeRobot writer keeps one `VideoWriter` state machine per `video_key`; run grouping stays synchronous and export work is serialized per key before episode metadata is applied back onto rows.
- `hydrate_media(...)` is intentionally narrow here; it is not a general file/bytes hydration helper.
