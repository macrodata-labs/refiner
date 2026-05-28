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

`fields` maps output frame columns to MCAP sources. A source can be a whole topic
or a decoded field path like `"/joint_states.position"`. If `fields` is omitted,
all selected non-video topics are preserved, and decoded object fields are
expanded into `topic.field` columns.

`primary` controls synchronization. When set, its timestamps define dense frame
rows and other fields/videos are nearest-aligned to those rows. When omitted,
the frame table is sparse over the union of selected message timestamps.

Set `messages_column="messages"` to also include the raw message table for
debugging.

## Related Pages

- [Reader Model](reader-model.md)
- [Files and Videos](files-and-videos.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
