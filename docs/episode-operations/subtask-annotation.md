---
title: "Subtask annotation"
description: "Use vision-language models to segment robotics episodes into temporal subtasks"
---

# Subtask annotation

Use `subtask_annotation` to add temporal subtask segments to robotics episodes,
then use `subtask_labeling` to label those fixed segments with stronger
previous/current/next visual context.
The annotator samples an episode video into timestamped contact sheets, sends
those images to a vision-language model, and writes the returned segments back
to each row.

Subtask labels are useful when training policies that need a denser signal than
the episode-level task description. They can identify coarse manipulation events
such as reaching, grasping, moving, opening, pouring, or placing objects without
manual per-episode labeling.

## Basic usage

Run the following pipeline locally or on the Macrodata Cloud:

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
    .map_async(
        mdr.robotics.subtask_labeling(
            video_key="observation.images.top",
            segments_column="predicted_subtasks",
            output_column="labeled_subtasks",
        ),
        max_in_flight=256,
    )
)
```

By default, `subtask_annotation` uses Gemini through `GoogleEndpointProvider`.
Set `GOOGLE_GENERATIVE_AI_API_KEY` before running the pipeline, or pass a
different provider explicitly. For Google providers, the segmentation and
labeling blocks pass Gemini `BLOCK_NONE` safety settings for harassment, hate
speech, sexually explicit, and dangerous-content categories to match the
benchmark contact-sheet runner and reduce false blocks on robotics videos.

The labeling block is optional, but it is the recommended default when you
want the best labels for fixed segments. If a segment already has a non-empty
`subtask`, `subtask_labeling` runs the seed-aware relabeling prompt. If
segments only have timestamps, it runs a plain labeling prompt from
previous/current/next visual context.

## Other readers

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

## Output shape

The segmentation output column contains a list of segment dictionaries. Each
segment has a start time, an end time, and a short action description under
`subtask`:

```python
[
    {"start_sec": 0.0, "end_sec": 1.2, "subtask": "reach object"},
    {"start_sec": 1.2, "end_sec": 2.8, "subtask": "grasp object"},
]
```

`subtask_labeling` writes fixed segments with a `subtask` value rewritten from a
dedicated labeling pass. If the input segment has a non-empty `subtask`, the
labeling prompt treats it as the seed label. If the input segment omits
`subtask` or sets it to an empty string, the block uses a plain labeling prompt.

```python
pipeline = pipeline.map_async(
    mdr.robotics.subtask_labeling(
        video_key="observation.images.top",
        segments_column="segments",
        output_column="labeled_subtasks",
    ),
    max_in_flight=256,
)
```

## Contact sheets

Video input is sent as contact sheets: timestamped image grids that preserve
temporal context without sending the full video. This keeps requests smaller
while giving the model enough visual context to choose event boundaries.

![Timestamped contact sheet](../assets/contact-sheet.png)

The default settings sample one frame every `0.5` seconds, resize each tile to
`224px` wide, and pack up to `20` frames per sheet in `5` columns.

For labeling, Refiner renders up to three fixed-segment contact sheets: the
previous segment, the current target segment, and the next segment. Each segment
sheet samples up to five timestamped frames, resizes each tile to `336px` wide
with Pillow `BOX` resampling, packs frames into `3` columns, and encodes JPEG
with quality `95` and subsampling `2`. Missing neighbors at episode boundaries
are represented by blank sheets. The model is instructed to label only the
current target segment and use the neighbors only to disambiguate what changed.

## Segmentation parameters

| Parameter | Meaning |
| --- | --- |
| `video_key` | Video stream to annotate. |
| `output_column` | Row column that receives the predicted segment list. |
| `sample_sec` | Seconds between sampled frames. |
| `frames_per_sheet` | Maximum number of sampled frames per contact sheet. |
| `on_blocked_prompt` | Behavior when the provider still blocks an episode prompt after the built-in fallback retry. Defaults to `"empty"`, which logs the block and writes an empty segment list. Use `"raise"` to fail the row instead. |
| `max_concurrent_requests` | Maximum provider requests allowed at once per worker. |

## Labeling parameters

| Parameter | Meaning |
| --- | --- |
| `video_key` | Video stream used to render previous/current/next segment sheets. |
| `segments_column` | Column containing fixed segment dictionaries with `start_sec`, `end_sec`, and optional seed `subtask`. Defaults to `predicted_subtasks`. |
| `output_column` | Row column that receives the relabeled segment list. Defaults to `labeled_subtasks`. Each output segment uses `subtask` for the relabeled text. |
| `on_blocked_prompt` | Behavior when the provider blocks a labeling prompt. Defaults to `"seed"`, which keeps the seed subtask. Use `"raise"` to fail the row. |
| `max_concurrent_requests` | Maximum provider requests allowed at once per worker. |

## Related content

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).

For a complete cloud example, see
[Video Subtask Annotations](../examples/annotations/subtask-annotations.md).
