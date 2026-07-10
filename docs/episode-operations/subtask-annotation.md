---
title: "Subtask annotation"
description: "Use vision-language models to segment robotics episodes into temporal subtasks"
---

# Subtask annotation

Use `subtask_annotation` to add temporal subtask segments to robotics episodes,
then use `subtask_labeling` to re-label those fixed segments with stronger
previous/current/next visual context.
The annotator samples an episode video into timestamped contact sheets, sends
those images to a vision-language model, and writes the returned segments back
to each row.

Subtask labels are useful when training policies that need a denser signal than
the episode-level task description. They can identify manipulation events such as
picking, opening, pouring, tool use, or placing objects without manual per-episode
labeling. The selected profile, rather than a hard-coded duration, determines the
event depth.

## Basic usage

Run the following pipeline locally or on the Macrodata Cloud:

```python
profile = mdr.robotics.DomainProfile(
    domain_id="acme-assembly",
    version="1",
    policy=mdr.robotics.MANIPULATION_EVENTS_V1,
    gold_set="acme-assembly-consensus-v1",
)

pipeline = (
    mdr.read_lerobot("hf://datasets/acme/demos")
    .map_async(
        mdr.robotics.subtask_annotation(
            profile=profile,
            video_key="observation.images.top",
            output_column="predicted_subtasks",
            result_column="subtask_annotation_result",
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
different provider explicitly.

The domain profile is required. It records the domain and the exact policy that
defines which events should become segments. Do not reuse a profile across domains
unless the domains share both the annotation policy and a validated model.

Refiner includes the frozen profiles used by the current evaluation program:

```python
walden_profile = mdr.robotics.WALDEN_V1
assembly_profile = mdr.robotics.ASSEMBLY_V1
```

`WALDEN_V1` pins the Walden-97 gold set and the Walden epoch-019 Partitioner.
`ASSEMBLY_V1` pins the assembly 40-by-5 consensus gold set. It intentionally has
no model artifact yet, so Refiner will reject attempts to use the Walden count
model with the assembly profile.

The labeling block is optional, but it is the recommended default when you
want the best labels for fixed segments. If a segment already has a non-empty
`subtask`, `subtask_labeling` will improve the labels using better conditioning. If
segments only have timestamps, it runs a plain labeling prompt from
previous/current/next visual context. If segmentation marked the segment column as
`None`, labeling propagates `None` without decoding the video or calling the model.

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
            profile=profile,
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

The result column contains the auditable sidecar for that list:

```python
{
    "status": "ok",  # ok, empty, partial, invalid, or blocked
    "segments": [...],
    "raw_segments": [...],
    "issues": [],
    "block_reason": None,
    "provenance": {
        "domain_id": "acme-assembly",
        "profile_version": "1",
        "policy_id": "manipulation-events",
        "policy_version": "1",
        "profile_hash": "...",
        "config_hash": "...",
        "backend": "vlm-contact-sheets",
        "model": "gemini-3.5-flash",
        "count_prior": None,
        "fallback_used": False,
        "latency_ms": 1234.5,
        "usage": {"prompt_tokens": 1200},
    },
}
```

`empty` means the model returned a valid empty timeline. `blocked` and `invalid`
write `None` to the segment output column, so provider or structural failures cannot
be confused with a video containing no manipulation events. Structural validation
clips timestamps to the video duration and reports overlaps, but does not snap,
merge, or otherwise change semantic boundaries.

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

## Count priors

Gemini can use a segment-count prior from an upstream, domain-matched partitioner:

```python
mdr.robotics.subtask_annotation(
    profile=profile,
    video_key="observation.images.top",
    count_prior_column="partitioner_segment_count",
)
```

The value must be a positive integer. It controls policy depth only; the prompt
still requires Gemini to place boundaries at visible semantic events. Do not use a
count produced by a model that has not been validated on the profile's domain.
`None` means the partitioner was unavailable or returned no segments, and runs the
annotation without a prior.

For already-computed Partitioner intervals, derive the count without another
network call:

```python
pipeline = pipeline.map(
    mdr.robotics.count_prior_from_segments(
        profile=mdr.robotics.WALDEN_V1,
        segments_column="walden_partitioner_segments",
        output_column="partitioner_segment_count",
    )
)
```

For the learned service, provide an episode-specific signed URL and data hash. The
adapter implements the learned `/segment` contract (`start_s`, `end_s`, and
`score`), not the separate VLM `/process` endpoint:

```python
pipeline = (
    pipeline
    .map_async(
        mdr.robotics.partitioner_count_prior(
            profile=mdr.robotics.WALDEN_V1,
            endpoint="https://partitioner.internal/segment",
            video_url_column="signed_video_url",
            data_hash_column="data_hash",
            token_env="REFINER_PARTITIONER_TOKEN",
        ),
        max_in_flight=8,
    )
    .map_async(
        mdr.robotics.subtask_annotation(
            profile=mdr.robotics.WALDEN_V1,
            video_key="observation.images.top",
            count_prior_column="partitioner_segment_count",
        )
    )
)
```

The bearer token is read inside the worker and is never captured in the pipeline
plan. Service failures write `None` for the count and an explicit `unavailable` or
`invalid` `partitioner_count_result`. Use `on_unavailable="raise"` when the prior
is mandatory.

## Evaluation

Use the same metrics for every model, prompt, and policy release:

```python
metrics = mdr.robotics.evaluate_subtask_segments(
    predicted_segments,
    consensus_segments,
    video_duration_s=42.5,
    iou_threshold=0.5,
    boundary_tolerance_s=0.5,
)
```

The result includes R@0.5/R@0.7, precision, recall, F1, mean IoU, boundary MAE,
oversegmentation ratio, boundary-F1, and edit cost per minute. Edit cost uses
`3 * creates + deletes + drags`; `drag_tolerance_s` defines when a matched start or
end boundary needs a drag. Use paired per-video comparisons rather than selecting
a model from aggregate R@0.5 alone.

For a full dataset, place records in JSON using `video_id`, `duration_s`, and
`segments` for gold; candidate files need `video_id` plus either `segments` or the
normal Refiner `predicted_subtasks`/`subtask_annotation_result` fields. Then run:

```bash
python -m refiner.robotics.subtask_annotation.benchmark_cli \
  --gold assembly-gold.json \
  --candidate current=current.json \
  --candidate v3p=v3p.json \
  --baseline current \
  --out assembly-report.json
```

The report contains per-video and aggregate metrics, missing/invalid/blocked rates,
latency and token means, SHA-256 input hashes, and paired bootstrap confidence
intervals. The default 2,000 resamples and seed are recorded in the report.

## Live provider smoke test

The repository includes an opt-in, billable test that renders a synthetic video,
sends a structured request to Gemini, and verifies status, segments, profile/config
hashes, count prior, and model provenance:

```bash
REFINER_RUN_LIVE_GEMINI=1 \
GOOGLE_GENERATIVE_AI_API_KEY=... \
REFINER_GEMINI_SMOKE_MODEL=gemini-3.5-flash \
REFINER_GEMINI_THINKING_BUDGET=16384 \
pytest tests/robotics/test_subtask_annotation_live.py -v
```

Set the model environment variable to the exact release under evaluation. The
requested model and thinking budget are incorporated into the config hash; the
test fails on blocked or structurally invalid output.

## Segmentation parameters

| Parameter | Meaning |
| --- | --- |
| `profile` | Required versioned `DomainProfile` containing the domain and segmentation policy. |
| `video_key` | Video stream to annotate. |
| `output_column` | Row column that receives the predicted segment list. |
| `result_column` | Row column that receives status, raw and validated segments, issues, and provenance. |
| `count_prior_column` | Optional row column containing a positive integer count from a domain-matched partitioner. |
| `thinking_budget` | Optional positive Google thinking-token budget, pinned in the request and config hash. |
| `on_blocked_prompt` | Behavior when both provider attempts are blocked. Defaults to `"mark"`, which writes an explicit blocked result and `None` segments. Use `"raise"` to fail the row instead. |
| `max_concurrent_requests` | Maximum provider requests allowed at once per worker. |

## Labeling parameters

| Parameter | Meaning |
| --- | --- |
| `video_key` | Video stream used to render previous/current/next segment sheets. |
| `segments_column` | Column containing fixed segment dictionaries with `start_sec`, `end_sec`, and optional seed `subtask`. Defaults to `predicted_subtasks`. |
| `output_column` | Row column that receives the relabeled segment list. Defaults to `labeled_subtasks`. Each output segment uses `subtask` for the relabeled text. |
| `on_blocked_prompt` | Behavior when the provider blocks a labeling prompt. Defaults to `"seed"`, which keeps the seed subtask. Use `"raise"` to fail the row. |
| `max_concurrent_segments` | Maximum labeling calls launched concurrently for one episode. Defaults to `8`; the provider-wide limit still applies. |
| `max_concurrent_requests` | Maximum provider requests allowed at once per worker. |

## Related content

For lower-level inference controls, see [Generate Text](../inference/generate-text.md)
and [Multimodal and Structured Output](../inference/multimodal-and-structured-output.md).

For a complete cloud example, see
[Video Subtask Annotations](../examples/annotations/subtask-annotations.md).

For background and design decisions, see
[Annotating Robot Video Subtasks](https://macrodata.co/blog/annotating-robot-video-subtasks).
