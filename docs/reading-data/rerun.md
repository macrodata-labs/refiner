---
title: "Rerun reader"
description: "Read Rerun RRD recordings as columnar or robotics episode rows"
---

# Rerun reader

Use `read_rerun` for `.rrd` files written by Rerun.

```python
import refiner as mdr

pipeline = mdr.read_rerun(
    "s3://bucket/run/**/*.rrd",
    output="robotics",
    fps=30,
)
```

Install `macrodata-refiner[rerun]` to use this reader. Add storage extras such
as `s3` when reading remote paths.

Directory inputs are filtered to paths ending in `.rrd`. RRD files are planned
as atomic files, so workers parallelize across recordings instead of splitting
one recording by byte range.

## Recording rows

With `output="recording"`, `read_rerun` emits one row per Rerun recording
segment:

```python
rows = mdr.read_rerun(
    "/data/run/*.rrd",
    output="recording",
    contents=("/action/**", "/observation/**"),
    timelines=("frame",),
)
```

Each row includes:

| Column | Meaning |
| --- | --- |
| `episode_id` | Rerun recording id or segment id. |
| `rerun` | `RerunRecording` value with Arrow-backed `Tabular` tables by timeline. |
| `file_path` | Source RRD path, unless `file_path_column=None`. |

`contents` is passed to Rerun's content filter. `timelines` limits the timeline
tables returned. If `timelines` is omitted, the reader materializes all timeline
indexes reported by the Rerun schema.

For raw RRD copy workflows that immediately call `write_rerun`, set
`materialize_tables=False`. The row still carries the source recording metadata
needed by the writer's chunk-copy path, but skips the Arrow timeline/static
tables that downstream code will not inspect.

## Robotics rows

With `output="robotics"`, the reader creates rows that can be passed to
`to_robot_rows(...)` and robotics writers:

```python
robot_rows = (
    mdr.read_rerun(
        "/data/episodes/*.rrd",
        output="robotics",
        fps=30,
        robot_type="unknown",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        nested_frames_key="frames",
        fps_key="fps",
        robot_type_key="robot_type",
        video_keys={
            "observation.images.top": "cam.top",
            "observation.images.left_wrist": "cam.left_wrist",
        },
    )
)
```

The default robotics mapping reads scalar components under `/action/**` into
the frame `action` vector, scalar components under `/observation/state/**` into
`observation.state`, and encoded images under `/cam/**` into top-level video
sources such as `cam.top`.

Use explicit selections when vector order or camera names matter:

```python
mdr.read_rerun(
    "episode.rrd",
    output="robotics",
    actions=("/robot/actions/gripper", "/robot/actions/arm"),
    states=("/robot/state/qpos", "/robot/state/gripper"),
    videos={"observation.images.top": "/robot/cameras/top"},
    fps=30,
)
```

`actions` and `states` define vector order. `videos` maps output video keys to
Rerun encoded-image entity paths. If `contents` is omitted, explicit selections
also define the minimal Rerun content filter for those categories.

## Decoding

Scalar action and state columns are read from Arrow list arrays and converted
to frame vectors. Encoded images remain lazy `VideoFrameSequence` values; JPEG
or PNG bytes are decoded frame-by-frame only when a downstream video writer or
consumer iterates the sequence.

## Sharding

Rerun SDK queries require a complete local RRD file. For that reason,
`read_rerun` sets file-atomic sharding, like other container readers such as
HDF5 and MCAP. `target_shard_bytes` groups whole RRD files into shard buckets,
and `num_shards` can request a target number of file buckets when there are
enough files.
