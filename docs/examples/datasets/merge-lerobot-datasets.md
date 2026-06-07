---
title: "Merge LeRobot Datasets"
description: "Combine multiple LeRobot roots into one output dataset"
---

# Merge LeRobot Datasets

Pass multiple LeRobot roots to `read_lerobot` and write one output dataset:

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot(
        [
            "hf://datasets/acme/pick-cubes-part-a",
            "hf://datasets/acme/pick-cubes-part-b",
        ]
    )
    .write_lerobot("hf://buckets/acme-robotics/pick-cubes-merged")
)

pipeline.launch_cloud(
    name="merge-pick-cubes",
    num_workers=4,
    refiner_extras=("hf", "video"),
    secrets={"HF_TOKEN": None},
)
```

The reader merges task metadata. The writer finalizes output metadata in a
reducer stage.
