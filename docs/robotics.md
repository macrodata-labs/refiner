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

## How Robotics Data Is Read

`read_lerobot(...)` yields one row per episode.

Those rows include:

- `frames`
- per-episode metadata
- video feature columns as handles
- dataset and episode stats metadata

So the programming model is the same as the rest of Refiner: you read rows,
transform rows, and then write rows through a sink. The difference is that the
rows are episode-oriented and the writer understands how to materialize a
LeRobot dataset back out.

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
pipeline = pipeline.map(lambda row: row.update(dataset_split="train"))
```

Because `map(...)` patches rows by default, this is a convenient way to add
episode-level annotations or derived fields before writing.

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
        num_workers=1,
    )
)
```

`motion_trim(...)` assumes LeRobot episode rows:

- it expects a `LeRobotRow`
- it trims the episode frame table directly
- it updates video timestamps on the row itself
- when a video span changes, the corresponding `stats/<video_key>/...` fields are dropped so the writer recomputes them later

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
        num_workers=1,
    )
)
```

## Related Pages

- [Readers and sharding](readers-and-sharding.md)
- [Pipeline basics](pipeline-basics.md)
- [Launchers](launchers.md)
- [Observability](observability.md)
