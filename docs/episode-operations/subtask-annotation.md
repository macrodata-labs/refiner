---
title: "Subtask Annotation"
description: "Use vision-language models to segment robotics episodes into temporal subtasks"
---

# Subtask Annotation

Use `subtask_annotation` to add temporal subtask labels to robotics episodes.
The annotator samples an episode video into timestamped contact sheets, sends
those images to a vision-language model, and writes the returned segments back
to each row.

Subtask labels are useful when training policies that need a denser signal than
the episode-level task description. They can identify coarse manipulation events
such as reaching, grasping, moving, opening, pouring, or placing objects without
manual per-episode labeling.

## Basic Usage

Run the following pipeline locally or on Refiner Cloud:

```python
pipeline = (
    mdr.read_lerobot("hf://datasets/acme/demos")
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key="observation.images.top",
            output_column="predicted_subtasks",
        ),
        max_in_flight=256,
    )
)
```

By default, `subtask_annotation` uses Gemini through `GoogleEndpointProvider`.
Set `GOOGLE_GENERATIVE_AI_API_KEY` before running the pipeline, or pass a
different provider explicitly.

## Other Readers

Input rows must implement `RoboticsRow` and expose the selected video through
`row.videos`. If you start from another reader, convert rows with
`to_robot_rows(...)` before annotation:

```python
pipeline = (
    mdr.read_parquet("/data/episodes.parquet")
    .to_robot_rows(
        episode_id_key="episode_id",
        task_key="tasks",
        video_keys={"observation.images.top": "video_uri"},
        fps=30.0,
    )
    .map_async(
        mdr.robotics.subtask_annotation(
            video_key="observation.images.top",
        ),
        max_in_flight=256,
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

Video input is sent as contact sheets: timestamped image grids that preserve
temporal context without sending the full video. This keeps requests smaller
while giving the model enough visual context to choose event boundaries.

![Timestamped contact sheet](../images/contact-sheet.png)

The default settings sample one frame every `0.5` seconds, resize each tile to
`224px` wide, and pack up to `20` frames per sheet in `5` columns.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `video_key` | Video stream to annotate. |
| `output_column` | Row column that receives the predicted segment list. |
| `sample_sec` | Seconds between sampled frames. |
| `frame_width` | Width of each sampled frame tile. |
| `frames_per_sheet` | Maximum number of sampled frames per contact sheet. |
| `columns` | Contact sheet grid columns. |
| `quality` | JPEG quality for generated sheet images, from `1` to `100`. Defaults to `84`. |
| `include_contact_sheet_manifest` | Add textual sheet descriptions to the prompt. |
| `min_segment_duration_sec` | Minimum returned segment duration. Defaults to `0.0`, so valid short segments are kept. |
| `on_blocked_prompt` | Behavior when the provider blocks an episode prompt. Defaults to `"empty"`, which logs the block and writes an empty segment list. Use `"raise"` to fail the row instead. |
| `max_concurrent_requests` | Maximum provider requests allowed at once per worker. |

## Related Content

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).

For a complete cloud example, see
[Video Subtask Annotations](../examples/annotations/subtask-annotations.md).
