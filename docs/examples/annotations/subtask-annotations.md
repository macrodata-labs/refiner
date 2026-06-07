---
title: "Annotate Subtasks"
description: "Add VLM-generated temporal subtask annotations to episodes"
---

# Annotate Subtasks

![Subtask annotation timeline](../../assets/subtask_annotations.png)

This example reads a LeRobot dataset, runs temporal subtask annotation on each
episode video, writes the predicted segments into a new `predicted_subtasks`
column, and saves the result back in LeRobot format.

```python
import refiner as mdr

INPUT_DATASET = "hf://datasets/lerobot/berkeley_cable_routing"
OUTPUT_ROOT = "hf://buckets/macrodata/test_bucket"
VIDEO_KEY = "observation.images.top_image"


pipeline = (
    mdr.read_lerobot(INPUT_DATASET)
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key=VIDEO_KEY,
            output_column="predicted_subtasks",
        ),
    )
    .write_lerobot(f"{OUTPUT_ROOT}/berkeley-cable-routing-subtasks")
)

pipeline.launch_cloud(
    name="berkeley-subtask-annotation",
    num_workers=1,
    cpus_per_worker=1,
    mem_mb_per_worker=2048,
    refiner_extras=("hf", "video"),
    secrets=mdr.Secrets.dict(
        {
            "HF_TOKEN": None,
            "GOOGLE_GENERATIVE_AI_API_KEY": None,
        }
    ),
)
```

`HF_TOKEN` and `GOOGLE_GENERATIVE_AI_API_KEY` are loaded from your local
environment at submission time because each value is set to `None`. Refiner
passes the resolved values to the Cloud job as redacted secrets. Export both
variables before launching the pipeline.

Use [Subtask Annotation](../../episode-operations/subtask-annotation.md) for
parameter details and in-depth explanation. `subtask_annotation` uses Gemini
3.5 Flash through `GoogleEndpointProvider`, so you need to provide
`GOOGLE_GENERATIVE_AI_API_KEY`.
