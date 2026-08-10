---
title: "Episode rows"
description: "The episode-level row model used by Refiner"
---

# Episode rows

Refiner exposes robotics data as rows. A row may be a plain mapping, a
`RoboticsRow` semantic view, or a `LeRobotRow`.

## Plain rows

Generic readers emit regular rows:

```python
row = mdr.read_parquet("/data/episodes.parquet").take(1)[0]
print(row["episode_id"])
```

Plain rows are enough for tabular transforms. Robotics operations usually need
a semantic episode view.

## `RoboticsRow`

`RoboticsRow` is a protocol: it describes what an episode row can do without
requiring one physical storage format.

Common properties:

| Property | Meaning |
| --- | --- |
| `episode_id` | Stable episode identifier. |
| `num_frames` | Number of frame rows. |
| `task` | Optional task text. |
| `fps` | Optional frame rate. |
| `robot_type` | Optional robot family/type. |
| `timestamps` | Frame-aligned timestamps. |
| `actions` | Frame-aligned actions. |
| `states` | Frame-aligned state observations. |
| `videos` | Mapping of video key to video source. |
| `stats` | Feature statistics when available. |

## `LeRobotRow`

`read_lerobot(...)` emits `LeRobotRow` values:

```python
row = mdr.read_lerobot("hf://datasets/lerobot/aloha_sim_transfer_cube_human").take(1)[0]

print(row.episode_index)
print(row.tasks)
print(row.metadata.info.fps)
```

`LeRobotRow` is also a `RoboticsRow`, so episode operations such as
[Motion Trimming](../episode-operations/motion-trimming.md) can use it directly.

## Updating rows

Rows are immutable-style values: update methods return new rows.

```python
def add_quality_flag(row):
    return row.update({"quality": "ok"})
```

Frame-aware rows also provide helpers:

```python
def zero_actions(row):
    return row.with_actions([[0.0] for _ in range(row.num_frames)])
```

## Selecting frames

```python
def first_second(row):
    count = int((row.fps or 30) * 1.0)
    return row.select_frames(range(min(count, row.num_frames)))
```

Selecting frames updates the frame table. For `LeRobotRow`, frame indices are
renumbered when present.

## Related pages

- [Frames and Videos](frames-and-videos.md)
- [Row Transforms](../transforms/row-transforms.md)
- [Writing LeRobot](../writing-data/lerobot.md)

