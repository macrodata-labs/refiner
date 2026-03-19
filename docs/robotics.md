---
title: "Robotics"
description: "Robotics data workflows in Refiner"
---

Refiner already includes robotics-specific support through the LeRobot reader,
writer, and robotics transforms.

## What You Can Do Today

- read LeRobot datasets with `read_lerobot(...)`
- transform episode rows with normal pipeline ops like `map(...)`, `filter(...)`, and expression-backed transforms where applicable
- run robotics-specific transforms under `mdr.robotics.*`
- write LeRobot-compatible output datasets with `write_lerobot(...)`
- merge compatible LeRobot datasets into one output dataset

## Quick Toc

- [reading datasets](#reading-datasets)
- [transforming rows](#transforming-rows)
- [writing datasets](#writing-datasets)
- [motion trimming](#motion-trimming)
- [performance notes](#lerobot-performance-notes)
- [merging datasets](#merging-datasets)

## Reading Datasets

Read a single LeRobot dataset:

```python
import refiner as mdr

pipeline = mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
```

Read multiple compatible datasets together:

```python
import refiner as mdr

pipeline = mdr.read_lerobot(
    [
        "hf://datasets/macrodata/aloha_static_battery_ep005_009",
        "hf://datasets/macrodata/aloha_static_battery_ep000_004",
    ]
)
```

## Transforming Rows

Once read, LeRobot data is manipulated through the same row pipeline model as
everything else in Refiner.

Example:

```python
pipeline = pipeline.map(lambda row: row.update(source_dataset="aloha_static_battery"))
```

Because `map(...)` patches rows by default, this is a convenient way to add episode-level metadata or derived fields before writing.

## Working With `LeRobotRow`

`read_lerobot(...)` yields `LeRobotRow` objects, so you usually do not need to
manually unpack the raw LeRobot transport fields.

`LeRobotRow` is a view over the underlying episode row, not a disconnected copy.
That means you get LeRobot-specific helpers on top, but you can still access and
patch the original row columns directly through the normal row API.

Typical things you will touch:

- `row.frames`
- `row.videos`
- `row.stats`
- `row.metadata`
- `row.update(...)`

Example:

```python
import refiner as mdr

def inspect_episode(row):
    assert isinstance(row, mdr.robotics.LeRobotRow)

    episode_index = row.episode_index
    length = row.length
    task_names = row.tasks
    fps = row.metadata.info.fps

    frames = row.frames
    first_frame = next(iter(frames))
    first_timestamp = first_frame["timestamp"]

    video_spans = {
        key: (video.from_timestamp_s, video.to_timestamp_s)
        for key, video in row.videos.items()
    }
    available_stats = list(row.stats)

    return row.update(
        debug_summary={
            "episode_index": episode_index,
            "length": length,
            "tasks": task_names,
            "fps": fps,
            "first_timestamp": first_timestamp,
            "videos": video_spans,
            "stats": available_stats,
        }
    )
```

The important unit here is:

- one pipeline row = one episode

not:

- one pipeline row = one frame

So transforms like `map(...)` and `motion_trim(...)` operate on an episode row
whose `frames` field contains that episode's frame table.

### Frames

`row.frames` is the episode frame payload.

In practice it is a normal `Tabular`, so you can:

- iterate it frame-by-frame
- inspect `row.frames.num_rows`
- access `row.frames.table` when you want the Arrow table
- replace it with `row.update(frames=...)`

Example:

```python
def keep_first_ten_frames(row):
    frames = row.frames
    kept = frames.with_table(frames.table.slice(0, 10))
    return row.update(frames=kept)
```

### Videos

`row.videos` is a mapping from video feature key to `LeRobotVideoRef`.

For each video ref, you can access:

- `video.uri`
- `video.from_timestamp_s`
- `video.to_timestamp_s`
- `video.video`
  - the underlying `VideoFile`

Example:

```python
def shift_videos_by_half_second(row):
    for key, video in row.videos.items():
        row = row.with_video(
            key,
            from_timestamp_s=(video.from_timestamp_s or 0.0) + 0.5,
            to_timestamp_s=(video.to_timestamp_s or 0.0) + 0.5,
        )
    return row
```

### Stats

`row.stats` is a LeRobot-aware mapping over `stats/<feature>/...` fields.

You can:

- iterate available feature names with `for feature in row.stats`
- read one feature with `row.stats["observation.images.main"]`
- drop stale stats with `row.stats.drop(feature)`

Example:

```python
def invalidate_video_stats(row):
    for key in row.videos:
        row = row.stats.drop(key)
    return row
```

### Metadata

`row.metadata` is a `LeRobotMetadata` dataclass carrying dataset-level state:

- `row.metadata.info`
- `row.metadata.stats`
- `row.metadata.tasks`

That is the right place to look for canonical dataset facts such as:

- `fps`
- `robot_type`
- merged task mapping

### Updating rows

Use:

- `row.update(...)` for normal row patches
- `row.with_video(...)` for video placement or timestamp patches
- `row.with_stats(...)` when you want to write one feature's stats back
- `row.drop(...)` to hide ordinary row fields

Those helpers return a new `LeRobotRow`; they do not mutate the input row.

## Writing Datasets

Use `write_lerobot(...)` to write a LeRobot-compatible output dataset:

```python
pipeline = pipeline.write_lerobot("hf://buckets/macrodata/my_robotics_output")
```

This is more than a generic file writer. The LeRobot writer handles:

- LeRobot-compatible dataset layout
- episode/frame materialization
- video handling
- dataset metadata reduction and finalization across stages

Current writer tuning is passed directly on `write_lerobot(...)`, including:

- `data_files_size_in_mb`
- `video_files_size_in_mb`
- `max_video_prepare_in_flight`
- `codec`
- `pix_fmt`
- `transencoding_threads`
- `encoder_options`
- `quantile_bins`
- `force_recompute_video_stats`

## LeRobot Performance Notes

Current LeRobot output is optimized for:

- incremental frame parquet writes
- asynchronous per-episode video preparation
- cheap metadata reduction after shard-local stage-1 output

Key decisions:

- remux is preferred when source packets and boundaries are compatible
- transcode is used when compatibility or stats recomputation requires decoded frames
- `max_video_prepare_in_flight` bounds concurrent episode video work inside one worker
- `transencoding_threads` is treated as a worker budget and divided across simultaneous video streams in the same row

The practical consequence is:

- frame-heavy no-video datasets mostly behave like a parquet writer
- video-heavy datasets are dominated by source probing, remux/transcode work, and the quality of source clip alignment

## Motion Trimming

Motion trimming is currently available through `mdr.robotics.motion_trim(...)`.

Example:

```python
import refiner as mdr

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(
        mdr.robotics.motion_trim(
            threshold=0.001,
            pad_frames=5,
        )
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_motion")
    .launch_cloud(
        name="motion_trim",
        num_workers=4,
    )
)
```

`motion_trim(...)` assumes LeRobot episode rows:

- it expects a `LeRobotRow`
- it trims the episode frame table directly
- it updates video timestamps on the row itself
- when a video span changes, the corresponding `stats/<video_key>/...` fields are dropped so the writer recomputes them later

## Merging Datasets

You can merge compatible LeRobot datasets by reading multiple roots and writing
them back through `write_lerobot(...)`.

```python
import refiner as mdr

(
    mdr.read_lerobot(
        [
            "hf://datasets/macrodata/aloha_static_battery_ep005_009",
            "hf://datasets/macrodata/aloha_static_battery_ep000_004",
        ]
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_merge")
    .launch_local(
        name="merge_aloha",
        num_workers=2,
    )
)
```

## Related Pages

- [Reading and writing data](reading-and-writing.md)
- [Pipeline basics](pipeline-basics.md)
- [Transforms](transforms.md)
- [Launchers](launchers.md)
- [Observability](observability.md)
