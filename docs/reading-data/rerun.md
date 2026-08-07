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

Set `materialize_tables=False` with `output="recording"` when you only need the
recording metadata and do not want to materialize Arrow timeline/static tables.
That mode keeps the row lightweight while preserving the source recording
identity and chunk metadata.

## Robotics rows

With `output="robotics"`, the reader creates robotics episode rows directly:

```python
robot_rows = mdr.read_rerun(
    "/data/episodes/*.rrd",
    output="robotics",
    fps=30,
    robot_type="unknown",
)
```

The default robotics mapping reads scalar components under `/action/**` into
the top-level `action` vector, scalar components under
`/observation/state/**` into top-level `observation.state`, and encoded images
under `/cam/**` into top-level video sources such as `cam.top`.

Rows also include a `rerun` recording sidecar by default. If `contents` is
omitted, that sidecar is built from the full recording view for the primary
timeline, not just the robotics prefixes. Project it away with `.drop("rerun")`
when downstream stages do not need the source Rerun structure.

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

Scalar action and state columns are read from Arrow list arrays and exposed as
top-level Arrow list arrays. Encoded images remain lazy `VideoFrameSequence`
values; JPEG or PNG bytes are decoded frame-by-frame only when a downstream
video writer or consumer iterates the sequence.

## Sharding

Rerun SDK queries require a complete local RRD file. For that reason,
`read_rerun` sets file-atomic sharding, like other container readers such as
HDF5 and MCAP. `target_shard_bytes` groups whole RRD files into shard buckets,
and `num_shards` can request a target number of file buckets when there are
enough files.
