---
title: "LeRobot Reader"
description: "Read LeRobot datasets as episode rows"
---

# LeRobot Reader

Use `read_lerobot` when your input is already a LeRobot dataset root.

```python
import refiner as mdr

pipeline = mdr.read_lerobot(
    "hf://datasets/lerobot/aloha_sim_transfer_cube_human",
)
```

Each output row is a `LeRobotRow`: one episode with metadata, frame data, and
video references attached.

## What Gets Loaded

`read_lerobot` reads:

| Dataset path | Used for |
| --- | --- |
| `meta/info.json` | Dataset fps, robot type, feature metadata, paths. |
| `meta/stats.json` | Existing feature statistics, when present. |
| `meta/tasks.parquet` | Task index to task text mapping. |
| `meta/episodes/...` | Episode-level rows. |
| `data/...` | Frame-level parquet slices referenced by episodes. |
| `videos/...` | Video files referenced lazily through row video views. |

Videos are not decoded when the dataset is read. They are decoded or
copied when a transform or writer needs frames.

## Multiple Inputs

Pass multiple roots to combine datasets:

```python
pipeline = mdr.read_lerobot(
    [
        "hf://datasets/acme/pick-cubes-a",
        "hf://datasets/acme/pick-cubes-b",
    ]
)
```

The reader merges metadata and remaps task indices so row tasks stay consistent.
For a full recipe, see [Merge LeRobot Datasets](../examples/datasets/merge-lerobot-datasets.md).

## Malformed Episodes

`read_lerobot` validates that each episode's declared frame count matches the
frames loaded from its frame parquet. Mismatches raise by default. To drop those
episodes instead, pass `skip_malformed_rows=True`; the reader warns once and
records `malformed_lerobot_episodes_skipped`.

## Inspecting Rows

```python
row = pipeline.take(1)[0]

print(row.episode_id)
print(row.tasks)
print(row.to_frame_table().names)
print(list(row.videos))
```

Read [Episode Rows](../episode-data/episode-rows.md) and
[Frames and Videos](../episode-data/frames-and-videos.md) for the row API.

## Sharding Options

```python
pipeline = mdr.read_lerobot(
    "hf://datasets/acme/large-demo-set",
    target_shard_bytes=256 * 1024 * 1024,
    num_shards=64,
)
```

The shard options apply to episode parquet files under `meta/episodes`.

## Related Pages

- [Writing LeRobot](../writing-data/lerobot.md)
- [Motion Trimming](../episode-operations/motion-trimming.md)
- [Subtask Annotation](../episode-operations/subtask-annotation.md)
