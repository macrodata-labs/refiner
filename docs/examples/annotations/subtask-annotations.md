---
title: "Annotate subtasks"
description: "Add VLM-generated temporal subtask annotations to episodes"
---

# Annotate subtasks

![Subtask annotation timeline](../../assets/subtask_annotations.png)

This example reads a LeRobot dataset, runs temporal subtask segmentation on each
episode video, optionally relabels the predicted subtasks to improve accuracy,
and saves the result back in LeRobot format.

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
    .map_async(
        mdr.robotics.subtask_labeling(
            video_key=VIDEO_KEY,
            segments_column="predicted_subtasks",
            output_column="labeled_subtasks",
        ),
    )
    .write_lerobot(f"{OUTPUT_ROOT}/berkeley-cable-routing-subtasks")
)

pipeline.launch_local(
    name="berkeley-subtask-annotation",
    num_workers=1,
)
```

Export `HF_TOKEN` and `GOOGLE_GENERATIVE_AI_API_KEY` before launching the
pipeline. Local workers inherit both values from your environment.

Use [Subtask Annotation](../../episode-operations/subtask-annotation.md) for
parameter details and in-depth explanation. Both `subtask_annotation` and
`subtask_labeling` use Gemini 3.5 Flash through `GoogleEndpointProvider`, so
you need to provide `GOOGLE_GENERATIVE_AI_API_KEY`.

For the benchmark context behind this workflow, see
[Annotating Robot Video Subtasks](https://macrodata.co/blog/annotating-robot-video-subtasks).

This is an open-source reference workflow. To have Macrodata evaluate subtask
annotations on representative episodes from your dataset,
[send us a sample](/contact).
