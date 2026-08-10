---
title: "Zarr writer"
description: "Write episode arrays and replay-buffer-style outputs to Zarr"
---

# Zarr writer

Use `write_zarr` when you want array-oriented output.

```python
pipeline.write_zarr(
    "/tmp/robot-buffer.zarr",
    arrays={
        "data/action": "action",
        "data/state": "observation.state",
    },
    episode_ends_path="meta/episode_ends",
)
```

For `RoboticsRow` inputs, Refiner can infer common robotics arrays when
`arrays` is omitted.

## Reducer

By default, `write_zarr` reduces shard-local stores into a single output store:

```python
pipeline.write_zarr(
    "/tmp/output.zarr",
    reduce_to_single_store=True,
)
```

Disable this only when you intentionally want per-shard stores.

## Related pages

- [Zarr Reader](../reading-data/zarr.md)
- [Media Assets and Reducers](media-assets-and-reducers.md)

