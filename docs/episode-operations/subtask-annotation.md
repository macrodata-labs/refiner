---
title: "Subtask Annotation"
description: "Use vision-language models to annotate temporal subtask segments"
---

# Subtask Annotation

Use `subtask_annotation` to add temporal action segments to LeRobot episodes.
The annotator samples the episode video into timestamped contact sheets, sends
those sheets to a vision-language model, and writes the returned segments back
onto each row.

This is useful when you want coarse manipulation events such as reaching,
grasping, moving, pouring, opening, closing, or placing objects without manually
labeling every episode.

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

The output column contains a list of segment dictionaries. Each segment has a
start time, an end time, and a short action description:

```python
[
    {"start_sec": 0.0, "end_sec": 1.2, "subtask": "reach object"},
    {"start_sec": 1.2, "end_sec": 2.8, "subtask": "grasp object"},
]
```

## Contact Sheets

Contact sheets reduce each video to a sequence of timestamped image grids. This
keeps requests smaller than full-video prompting while preserving enough visual
context for the model to choose event boundaries.

| Parameter | Meaning |
| --- | --- |
| `sample_sec` | Seconds between sampled frames. |
| `frame_width` | Width of each sampled frame in the sheet. |
| `frames_per_sheet` | Number of frames per sheet image. |
| `columns` | Contact sheet grid columns. |
| `include_contact_sheet_manifest` | Add textual sheet descriptions to the prompt. |
| `min_segment_duration_sec` | Minimum returned segment duration. Defaults to `0.0`, so valid short segments are kept. |

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).
