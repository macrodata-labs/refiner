---
title: "Metadata, tasks, and stats"
description: "LeRobot metadata, task labels, feature stats, and when they change"
---

# Metadata, tasks, and stats

LeRobot datasets carry metadata beside episode and frame data. Refiner reads and
writes that metadata so downstream training code can understand the dataset.

## Metadata

```python
row = mdr.read_lerobot("hf://datasets/acme/pick-cubes").take(1)[0]

print(row.metadata.info.fps)
print(row.metadata.info.robot_type)
print(row.metadata.info.features)
```

Metadata includes dataset-level information such as fps, robot type, feature
definitions, paths, and video encoding information.

## Tasks

LeRobot stores task text in a task table and references tasks by `task_index`.

```python
print(row.tasks)
print(row.task)
```

`row.task` is a convenience alias for the first item in `row.tasks`.

When reading multiple LeRobot roots, Refiner merges task tables and remaps
indices so task text stays stable.

## Stats

Feature stats describe distributions used by training and normalization code.

```python
if "action" in row.stats:
    print(row.stats["action"].mean)
```

Stats are preserved when they are still valid. Operations that change a feature
should drop stale stats for that feature. For example, motion trimming drops
video stats for clipped videos so the writer recomputes them.

## Writer behavior

`write_lerobot` writes:

| Output | Source |
| --- | --- |
| `meta/info.json` | merged feature metadata and output paths |
| `meta/tasks.parquet` | merged task table |
| `meta/stats.json` | computed or reused feature stats |
| `meta/episodes/...` | episode rows |
| `data/...` | frame parquet files |
| `videos/...` | video files |

See [Writing LeRobot](../writing-data/lerobot.md).

## Related pages

- [LeRobot Reader](../reading-data/lerobot.md)
- [LeRobot Writer](../writing-data/lerobot.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)
