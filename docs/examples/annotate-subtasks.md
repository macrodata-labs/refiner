---
title: "Annotate Subtasks"
description: "Add VLM-generated temporal subtask annotations to episodes"
---

# Annotate Subtasks

This example reads a LeRobot dataset, runs temporal subtask annotation on the
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
    num_workers=4,
    cpus_per_worker=1,
    mem_mb_per_worker=2048,
    secrets=[
        mdr.Secrets.env(keys=["HF_TOKEN"]),
        {"GOOGLE_GENERATIVE_AI_API_KEY": None},
    ],
)
```

`HF_TOKEN` is passed through from the default Macrodata Cloud environment so the
workers can read and write Hugging Face datasets and buckets without embedding
the token in the script. `GOOGLE_GENERATIVE_AI_API_KEY` is declared as a secret
that must be supplied by the launch environment because `subtask_annotation`
uses Gemini 3.5 Flash through `GoogleEndpointProvider`.

Use [Subtask Annotation](../episode-operations/subtask-annotation.md) for
parameter details. `subtask_annotation` uses Gemini 3.5 Flash through
`GoogleEndpointProvider`, so you need to provide `GOOGLE_GENERATIVE_AI_API_KEY`.
