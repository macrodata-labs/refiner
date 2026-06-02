---
title: "Subtask Annotation"
description: "Use vision-language models to annotate temporal subtask segments"
---

# Subtask Annotation

`subtask_annotation` samples contact sheets from a `RoboticsRow` episode video
and asks a vision-language model to return temporal subtask segments.

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/demos")
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key="observation.images.top",
            output_column="predicted_subtasks",
        ),
    )
)
```

Input rows must implement `RoboticsRow` and expose at least one video through
`row.videos`. `read_lerobot(...)` rows already satisfy this. For other readers,
call `to_robot_rows(...)` first and map the video/task fields:

```python
pipeline = (
    mdr.read_parquet("/data/episodes.parquet")
    .to_robot_rows(
        episode_id_key="episode_id",
        task_key="tasks",
        video_keys={"observation.images.top": "video_uri"},
        fps=30.0,
    )
    .map_async(mdr.robotics.subtask_annotation())
)
```

`subtask_annotation` uses Gemini 3.5 Flash through `GoogleEndpointProvider`,
so you need to provide `GOOGLE_GENERATIVE_AI_API_KEY`.

## Output Shape

The output column contains a list of segments:

```python
[
    {"start_sec": 0.0, "end_sec": 1.2, "subtask": "reach object"},
    {"start_sec": 1.2, "end_sec": 2.8, "subtask": "grasp object"},
]
```

## Contact Sheets

Contact sheets reduce video into timestamped image grids. We found this to be
the most efficient way to give VLMs temporal context for subtask annotation.

| Parameter | Meaning |
| --- | --- |
| `sample_sec` | Seconds between sampled frames. |
| `frame_width` | Width of each sampled frame in the sheet. |
| `frames_per_sheet` | Number of frames per sheet image. |
| `columns` | Contact sheet grid columns. |
| `quality` | JPEG quality for generated sheet images, from `1` to `100`. Defaults to `84`. |
| `include_contact_sheet_manifest` | Add textual sheet descriptions to the prompt. |
| `min_segment_duration_sec` | Minimum returned segment duration. Defaults to `0.0`, so valid short segments are kept. |

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).
