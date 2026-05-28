---
title: "MCAP Reader"
description: "Read MCAP robotics logs as episode rows"
---

# MCAP Reader

Use `read_mcap` for robotics or autonomy logs stored as MCAP files.

```python
import refiner as mdr

pipeline = (
    mdr.read_mcap(
        "/data/logs/*.mcap",
        fields={
            "state": "/joint_states.position",
            "action": "/joint_states.velocity",
        },
        videos={"front": "/camera/image/compressed"},
        primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="frames",
        state_key="state",
        action_key="action",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
)
```

Install `macrodata-refiner[mcap]` to use this reader.

## Output Rows

`read_mcap` emits one row per episode. By default, each input file is one
episode. `episode_splitting={"time_gap_s": seconds}` splits on timestamp gaps,
and `episode_splitting={"marker_topic": topic}` splits at marker messages.

Each row includes:

| Column | Meaning |
| --- | --- |
| `file_path` | Source MCAP file path. |
| `episode_index` | Episode number within the file. |
| `frames` | `Tabular` frame table. |
| `videos` | Mapping of selected video names to `VideoFrameArray` values, when `videos` is set. |
| `fps` | Explicit fps, or inferred fps when possible. |
| `message_count` | Number of selected messages in the episode. |
| `topics` | Sorted selected topic names present in the episode. |

## Selecting Topics And Fields

`topics` limits which MCAP topics are read. If omitted, the reader scans all
topics needed by `fields` and `videos`. If `topics` is set, include the data
topics used by `fields` and `videos`; the reader may still include control
topics internally when an unselected `primary` source or marker-based splitting
needs them.

`fields` maps output frame-table columns to MCAP sources:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "target": "/cmd.target",
    },
)
```

A source can be a whole topic, such as `"/joint_states"`, or a decoded subfield,
such as `"/joint_states.position"`. The reader first checks whether the exact
source string is a topic. If not, it treats the longest matching topic prefix as
the topic and the remainder as a dotted field path.

If `fields` is omitted, decoded object messages are expanded into default
columns like `"/joint_states.position"`. Selected video topics and marker topics
are excluded from those default frame columns.

## Episode Splitting

`read_mcap` always emits episode rows. By default, each input MCAP file becomes
one episode:

```python
mdr.read_mcap("run.mcap")
```

Use `episode_splitting={"time_gap_s": seconds}` when episodes are separated by a
long pause in the log:

```python
mdr.read_mcap(
    "run.mcap",
    episode_splitting={"time_gap_s": 5.0},
)
```

The reader gathers selected message timestamps, sorts them by log time, and
starts a new episode whenever the gap between consecutive timestamps is greater
than the threshold. The split is based on MCAP log timestamps, not file order.

Use `episode_splitting={"marker_topic": topic}` when the log contains explicit
episode markers:

```python
mdr.read_mcap(
    "run.mcap",
    fields={"state": "/state.q"},
    episode_splitting={"marker_topic": "/episode_start"},
)
```

Each marker timestamp starts an episode. The episode ends at the next marker, or
at the end of the file for the final marker. Marker messages are used only for
splitting and are not included as default frame columns.

If there are no messages, no time gaps, or no marker messages, the reader falls
back to one episode for the file.

## Synchronization

MCAP logs usually contain multiple topics with different rates: robot state,
actions, camera frames, commands, diagnostics, and events may not share exactly
the same timestamps. `read_mcap` supports two frame-table modes.

Without `primary`, the frame table is sparse:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "command": "/cmd.target",
    },
)
```

Sparse mode creates rows over the union of selected message timestamps. A column
is null when that field has no message at that timestamp. If multiple messages
share the same timestamp, duplicate rows are preserved.

With `primary`, the frame table is dense on the primary source:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "command": "/cmd.target",
    },
    primary="state",
)
```

The primary source can be an output field name (`"state"`), a video name, a topic,
or a dotted source path. Its timestamps define the output rows. Every other
field is nearest-neighbor aligned to those timestamps. If `include_skew=True`
`read_mcap` adds columns such as `mcap.command.timestamp` and
`mcap.command.skew_ms` so you can inspect how far each aligned sample was from
the primary timestamp.

The reader uses streaming MCAP reads, so raw events are not assumed to arrive in
timestamp order. It sorts timestamps where ordering affects semantics: splitting,
sparse frame rows, primary alignment, and fps inference.

## Videos

`videos` maps video names to MCAP sources:

```python
mdr.read_mcap(
    "run.mcap",
    videos={"front": "/camera/image/compressed"},
    primary="front",
    fps=30,
)
```

Each selected video becomes a `VideoFrameArray` under the row's `videos` column.
When you later call `to_robot_rows`, map those values into semantic video keys:

```python
pipeline = (
    mdr.read_mcap(
        "run.mcap",
        fields={"state": "/joint_states.position"},
        videos={"front": "/camera/image/compressed"},
        primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="frames",
        state_key="state",
        action_key=None,
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
)
```

Video sources must resolve to image-like frame payloads: ROS compressed image
messages, raw ROS image messages, decoded image arrays, or base64 image bytes.
The reader does not decode encoded video packet streams such as H.264 byte
streams. For those logs, read the packet metadata as fields, or decode the video
stream before converting it to robotics rows.

When `primary` is set, videos are nearest-aligned to the primary timestamps just
like regular fields. When `primary` is omitted, each video keeps the frames from
its own topic timestamps.

## Related Pages

- [Reader Model](reader-model.md)
- [Files and Videos](files-and-videos.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
