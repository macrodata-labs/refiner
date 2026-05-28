---
title: "Writing Data"
description: "Write robotics datasets and pipeline outputs from Refiner"
---

# Writing Data

Writers attach a sink to the end of a pipeline. Launched execution runs the
reader, transforms, and writer stages.

| Writer | Use it for |
| --- | --- |
| [LeRobot](lerobot.md) | Training-ready robotics datasets. |
| [Zarr](zarr.md) | Array stores and replay buffers. |
| [Parquet and JSONL](parquet-and-jsonl.md) | Tabular outputs and logs. |
| [Media Assets and Reducers](media-assets-and-reducers.md) | Asset uploads, video handling, and reducer stages. |

## Writer Pattern

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/raw")
    .map(mdr.robotics.motion_trim())
    .write_lerobot("hf://buckets/acme-robotics/trimmed")
)
```

The writer does work when the pipeline is launched.

