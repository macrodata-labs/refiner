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
  - `decode` can be `True`, `False`, or `None`:
    - `None` rejects timestamped video rows.
    - `False` keeps timestamped `Video` references without decoding.
    - `True` is reserved for clip-aligned materialization flows.
  - optional `limit` bounds emitted episodes per reader instance.
- Use `target_shard_bytes` to control shard granularity.

## Hydrating External Files

Use `refiner.hydrate_file(...)` in `.flat_map(...)` for buffered row-level hydration.

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("s3://bucket/dataset")
    .flat_map(mdr.hydrate_file(columns="observation.images.main"))
)
```

Hydration defaults to attaching a lazy `VideoFile` handle on `Video.file` (or replacing string URI fields with raw bytes).

If you explicitly need full in-memory bytes for videos, use `video_hydration="bytes"`:

```python
pipeline = pipeline.flat_map(
    mdr.hydrate_file(
        columns="observation.images.main",
        video_hydration="bytes",
    )
)
```
Set `max_in_flight` to control concurrency/backpressure. Rows are yielded in input order.

## LeRobot Function Adapters

Use `convert_le_robot_fc(...)` to adapt a LeRobot-style episode dict function into a normal `.map(...)` step:

```python
import refiner as mdr

def tweak_episode(ep: dict) -> dict:
    ep["episode_index"] += 1000
    return ep

pipeline = mdr.read_lerobot("s3://bucket/dataset").map(
    mdr.convert_le_robot_fc(tweak_episode)
)
```

## Internal Notes

- Parquet byte-lazy mode maps planned byte ranges to row groups at read time.
- Parquet row access uses batch-level cached column-name indexing for faster key lookup.
- LeRobot expects parquet metadata under `meta/episodes/**`; legacy JSONL metadata is not used.
- LeRobot reads `fps`, `robot_type`, `features`, `data_path`, and `video_path` from `meta/info.json` when present.
