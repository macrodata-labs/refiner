---
title: "Zarr Reader"
description: "Read Zarr replay buffers and episode ranges"
---

# Zarr Reader

Use `read_zarr` for replay buffers stored as Zarr groups or zipped Zarr stores.

## Episode Ranges

Many robotics replay buffers store frame-aligned arrays and cumulative episode
ends:

```python
import refiner as mdr

pipeline = (
    mdr.read_zarr(
        "hf://datasets/acme/robot_buffer/buffer.zarr",
        arrays={
            "action": "data/action",
            "eef_pos": "data/eef_pos",
            "joint_pos": "data/joint_pos",
            "wrist": "data/wrist_rgb",
        },
        row_ends="meta/episode_ends",
        index_column="episode_id",
    )
    .to_robot_rows(
        episode_id_key="episode_id",
        action_key="action",
        state_key=("eef_pos", "joint_pos"),
        video_keys=("wrist",),
    )
)
```

With `row_ends`, each emitted row contains one episode range. Refiner never
splits a row across episode boundaries.

## Leading-Axis Rows

For arrays where each leading-axis item is one row:

```python
pipeline = mdr.read_zarr(
    "/data/features.zarr",
    arrays={"embedding": "embeddings", "score": "scores"},
    split_leading_axis=True,
)
```

Use this for non-episode arrays or precomputed feature tables.

## Selected Arrays And Attributes

`arrays` and `attrs` can be mappings from output column name to Zarr path:

```python
pipeline = mdr.read_zarr(
    "/data/run.zarr",
    arrays={"action": "data/action"},
    attrs={"task": "meta/task"},
)
```

## Related Pages

- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
- [Zarr Replay Buffer Example](../examples/zarr-replay-buffer.md)
- [Writing Zarr](../writing-data/zarr.md)

