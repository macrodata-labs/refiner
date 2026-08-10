---
title: "Merge LeRobot datasets"
description: "Combine multiple LeRobot roots into one output dataset"
---

# Merge LeRobot datasets

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

pipeline.launch_local(
    name="merge-pick-cubes",
    num_workers=2,
)
```

The reader merges task metadata. The writer finalizes output metadata in a
reducer stage. Set `HF_TOKEN` in your local environment before launching so the
workers can read and write the Hugging Face datasets.
