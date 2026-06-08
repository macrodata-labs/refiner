---
title: "Media assets and reducers"
description: "How Refiner writers handle media files, assets, and reducer stages"
---

# Media assets and reducers

Robotics datasets often contain large media values. Refiner distinguishes the
row value from the asset storage behavior through dtypes and video source APIs.

## Asset columns

Use dtypes to mark media columns:

```python
pipeline = mdr.read_parquet(
    "/data/videos.parquet",
    dtypes={"video": mdr.datatype.video_path()},
)
```

Writers can then copy or upload assets instead of treating them as plain strings.

## Missing asset policy

Parquet and JSONL writers support `missing_asset_policy`:

```python
pipeline.write_parquet(
    "/tmp/out",
    upload_assets=True,
    missing_asset_policy="error",
)
```

Use `"error"` for training data. Missing media should usually fail the job.

## Reducers

Some writers add a reducer stage. A reducer stage finalizes outputs after
workers finish shard-local writes.

| Writer | Reducer purpose |
| --- | --- |
| LeRobot | Merge metadata, tasks, stats, and staged chunks. |
| Zarr | Merge shard-local stores into a single store when configured. |

Reducers are part of the launched pipeline plan and are visible in job progress.

