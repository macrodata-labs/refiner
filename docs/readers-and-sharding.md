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
  - `metadata`: dict with `lerobot_info` and `lerobot_stats`.
  - raw transport metadata keys under `stats/*`, `videos/*`, and `meta/episodes/*` are omitted from emitted rows.
  - frame slicing requires `dataset_from_index`/`dataset_to_index` in each episode row.
  - optional `limit` bounds emitted episodes per reader instance.
  - `media_max_in_flight` controls concurrent episode materialization work inside the reader.
  - `media_preserve_order` controls whether those async reader results are yielded in input order.
- Use `target_shard_bytes` to control shard granularity.

## Hydrating External Files

Use `refiner.hydrate_media(...)` with `.map_async(...)` when you want decoded
video clips materialized in-process.

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("s3://bucket/dataset")
    .map_async(
        mdr.hydrate_media("observation.images.main", decode=True),
        max_in_flight=8,
    )
)
```

`hydrate_media(...)` currently accepts `Video` values only. For LeRobot inputs,
that means clip-aligned hydration is explicit and decode-backed:

- pass `decode=True`
- use `.map_async(...)` to control concurrency
- expect decoded frames in `video.media`

Rows are yielded in input order unless you explicitly choose otherwise on the async step.

## LeRobot Writer Tuning

`write_lerobot(...)` keeps video threading explicit:

- `video_encoder_threads=<int>` and `video_decoder_threads=<int>` force exact FFmpeg thread counts.
- `video_encoder_threads=None` and `video_decoder_threads=None` auto-resolve once per shard from the number of video features and the worker CPU budget.
- More concurrent video tracks per row generally means fewer threads per track.

## Internal Notes

- Parquet byte-lazy mode maps planned byte ranges to row groups at read time.
- Parquet row access uses batch-level cached column-name indexing for faster key lookup.
- LeRobot expects parquet metadata under `meta/episodes/**`; legacy JSONL metadata is not used.
- LeRobot reads `fps`, `robot_type`, `features`, `data_path`, and `video_path` from `meta/info.json` when present.
- `hydrate_media(...)` is intentionally narrow here; it is not a general file/bytes hydration helper.
