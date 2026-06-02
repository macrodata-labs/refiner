---
title: "LeRobot Writer"
description: "Write Refiner episode rows as LeRobot datasets"
---

# LeRobot Writer

Use `write_lerobot` to write `LeRobotRow` or generic `RoboticsRow` values as a
LeRobot dataset.

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/raw")
    .map(mdr.robotics.motion_trim(threshold=0.001))
    .write_lerobot("hf://buckets/acme-robotics/trimmed")
)
```

## Input Requirements

Rows must be one of:

| Row type | How to get it |
| --- | --- |
| `LeRobotRow` | `read_lerobot(...)` |
| `RoboticsRow` | `to_robot_rows(...)` from HDF5, Zarr, files, or custom readers |

Plain rows should be adapted before writing; see
[Converting to Robot Rows](../episode-data/converting-to-robot-rows.md).
If a generic `RoboticsRow` has no episode id, the writer generates one from the
shard id and row position.

## Output Layout

The writer creates LeRobot-compatible data:

| Output path | Content |
| --- | --- |
| `data/chunk-.../file-...parquet` | Frame rows. |
| `videos/<key>/chunk-.../file-...mp4` | Video streams when rows include videos. |
| `meta/episodes/...` | Episode rows. |
| `meta/tasks.parquet` | Task table. |
| `meta/info.json` | Dataset metadata. |
| `meta/stats.json` | Feature statistics after reducer finalization. |

## Video Behavior

The writer can remux compatible clipped videos instead of decoding and
re-encoding them. It transcodes when required by format, clipping, stale stats,
or writer options.

```python
pipeline.write_lerobot(
    "hf://buckets/acme-robotics/output",
)
```

## Performance Knobs

| Option | Use it for |
| --- | --- |
| `data_files_size_in_mb` | Target frame parquet file size. |
| `video_files_size_in_mb` | Target video file size. |
| `codec` | Video codec for transcoded videos. Defaults to `mpeg4`. |
| `pix_fmt` | Pixel format for transcoded videos. Defaults to `yuv420p`. |
| `max_video_prepare_in_flight` | Bound concurrent episode video preparation per worker. Defaults to `4`. |
| `quantile_bins` | Accuracy/cost tradeoff for video stats quantiles. |
| `force_recompute_video_stats` | Recompute stats even when existing stats could be reused. |

## Related Pages

- [Metadata, Tasks, and Stats](../episode-data/metadata-tasks-and-stats.md)
- [Media Assets and Reducers](media-assets-and-reducers.md)
- [Merge LeRobot Datasets](../examples/merge-lerobot-datasets.md)
