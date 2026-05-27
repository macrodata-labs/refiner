---
title: "Motion Trimming"
description: "Trim inactive frames from robotics episodes"
---

# Motion Trimming

`motion_trim` removes inactive frames before and after the motion window of an
episode. It uses action and state deltas to infer activity.

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/raw-aloha")
    .map(mdr.robotics.motion_trim(threshold=0.001, pad_frames=2))
    .write_lerobot("hf://buckets/acme-robotics/aloha-trimmed")
)
```

## What It Updates

| Data | Behavior |
| --- | --- |
| Frame table | Keeps only selected frame indices. |
| Timestamps | Shifts kept timestamps so the trimmed episode starts at zero. |
| Videos | Clips video views to the same time span. |
| Video stats | Drops stale video stats so the writer recomputes them. |
| Metrics | Logs frames in/out, removed frames, and trim fraction when launched. |

## Parameters

| Parameter | Meaning |
| --- | --- |
| `threshold` | Motion energy threshold for action/state deltas. |
| `pad_frames` | Extra frames to keep before and after detected motion. |
| `timestamp_key` | Frame timestamp key when semantic timestamps are absent. |
| `action_key` | Frame action key when semantic actions are absent. |
| `state_key` | Frame state key when semantic states are absent. |

## Requirements

Input rows must implement `RoboticsRow` and contain non-empty frame data with
timestamps, actions, and state values. `read_lerobot(...)` rows satisfy this
when the dataset contains those fields. Generic rows should be converted with
[`to_robot_rows`](../episode-data/converting-to-robot-rows.md).

## Related Pages

- [Frames and Videos](../episode-data/frames-and-videos.md)
- [Writing LeRobot](../writing-data/lerobot.md)

