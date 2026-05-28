---
title: "Subtask Annotation"
description: "Use vision-language models to annotate temporal subtask segments"
---

# Subtask Annotation

`subtask_annotation` samples contact sheets from an episode video and asks a
vision-language model to return temporal subtask segments.

```python
provider = mdr.inference.VLLMProvider(model="Qwen/Qwen2.5-VL-7B-Instruct")

pipeline = (
    mdr.read_lerobot("hf://datasets/acme/demos")
    .map_async(
        mdr.robotics.subtask_annotation(
            provider=provider,
            video_key="observation.images.top",
            output_column="predicted_subtasks",
            sample_sec=0.5,
        ),
        max_in_flight=64,
    )
)
```

## Output Shape

The output column contains a list of segments:

```python
[
    {"start_sec": 0.0, "end_sec": 1.2, "subtask": "reach object"},
    {"start_sec": 1.2, "end_sec": 2.8, "subtask": "grasp object"},
]
```

## Contact Sheets

Contact sheets reduce video into timestamped image grids. This is often cheaper
and easier for VLMs than sending the full video. By default, the annotator
samples every `0.5` seconds, resizes each sampled frame to `224px` wide, and
packs frames chronologically into `5` columns by `4` rows. The default prompt
instructs the model to read time left-to-right, then top-to-bottom, and to use
the visible timestamp printed inside each tile when choosing segment boundaries.

| Parameter | Meaning |
| --- | --- |
| `sample_sec` | Seconds between sampled frames. |
| `frame_width` | Width of each sampled frame in the sheet. |
| `frames_per_sheet` | Number of frames per sheet image. |
| `columns` | Contact sheet grid columns. |
| `include_contact_sheet_manifest` | Add textual sheet descriptions to the prompt. |
| `min_segment_duration_sec` | Minimum returned segment duration. Defaults to `0.0`, so valid short segments are kept. |

## Prompting

Pass a custom prompt when you have a fixed subtask vocabulary:

```python
pipeline = pipeline.map_async(
    mdr.robotics.subtask_annotation(
        provider=provider,
        prompt="Label each episode with reach, grasp, move, and place segments.",
    ),
    max_in_flight=32,
)
```

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).
