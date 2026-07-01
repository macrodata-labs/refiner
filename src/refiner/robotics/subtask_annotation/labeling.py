from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.robotics.subtask_annotation.utils import (
    GEMINI_BLOCK_NONE_SAFETY_SETTINGS,
    TimestampedContactSheet,
    _blank_contact_sheet,
    _blocked_prompt_reason,
    _normalize_input_segments,
    _normalize_label,
    _resolve_video,
    _segment_contact_sheets,
)
from refiner.worker.context import logger

if TYPE_CHECKING:
    from refiner.inference.generate_text import GenerateTextFn


_SUBTASK_LABELING_PROMPT_TEMPLATE = """Annotate one fixed segment from a longer video.

Return only JSON:
{{"subtask":"short descriptive subtask label"}}

Inputs:
- The first image is the previous fixed segment, if it exists; otherwise it is blank/context only.
- The second image is the current target segment.
- The third image is the next fixed segment, if it exists; otherwise it is blank/context only.
- Each image is timestamped with absolute video time.

Episode instruction:
{instruction}

Target segment:
{segment_index} of {segment_count}

Target time:
{start_sec:.2f}s to {end_sec:.2f}s

Rules:
- Label only the current target segment.
- Use previous/next images only to disambiguate what changed during the current segment.
- Do not describe the previous or next segment.
- The segment boundaries are fixed; do not split or merge.
- Compare the beginning and end of the current segment and describe the visible manipulation or state change.
- Use one concise imperative phrase.
- Include the exact action and manipulated object.
- Include source, destination, side, direction, final location, opened/closed/filled/cleaned state, or affected part when visible and central.
- Do not mention timestamps, frame numbers, uncertainty, candidates, or invisible intent.
"""

_SUBTASK_SEED_LABELING_PROMPT_TEMPLATE = """Annotate one fixed segment from a longer video.

Return only JSON:
{{"subtask":"short descriptive subtask label"}}

Inputs:
- The first image is the previous fixed segment, if it exists; otherwise it is blank/context only.
- The second image is the current target segment.
- The third image is the next fixed segment, if it exists; otherwise it is blank/context only.
- Each image is timestamped with absolute video time.

Episode instruction:
{instruction}

Target segment:
{segment_index} of {segment_count}

Target time:
{start_sec:.2f}s to {end_sec:.2f}s

Original predicted label for this exact segment:
{seed_label}

Rules:
- Label only the current target segment.
- Use previous/next images only to disambiguate what changed during the current segment.
- Treat the original predicted label as a strong prior, not as ground truth.
- Verify and minimally correct the original label using the current target segment.
- If the original label describes the same action and main object, keep it, only improving grammar or adding clearly visible essential details.
- If it is too vague but directionally correct, make it more specific.
- If it describes the previous/next segment, the wrong action, wrong object, wrong destination, or wrong state change, replace it.
- Do not describe the previous or next segment.
- Do not split or merge the fixed segment.
- Do not introduce a new action unless it is clearly visible in the current target segment.
- Do not make the label broader than the fixed segment.
- Use one concise imperative phrase.
- Include the exact action and manipulated object.
- Include source, destination, side, direction, final location, opened/closed/filled/cleaned state, or affected part when visible and central.
- Do not mention timestamps, frame numbers, uncertainty, candidates, or invisible intent.
"""


class _SubtaskLabelingResult(BaseModel):
    subtask: str


def subtask_labeling(
    *,
    provider: InferenceProvider = GoogleEndpointProvider(model="gemini-3.5-flash"),
    video_key: str,
    segments_column: str = "predicted_subtasks",
    output_column: str = "labeled_subtasks",
    frame_width: int = 336,
    max_frames_per_segment: int = 5,
    columns: int = 3,
    quality: int = 95,
    temperature: float = 0.0,
    on_blocked_prompt: Literal["seed", "raise"] = "seed",
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that labels fixed subtask segments.

    If no seed subtask is available for a segment, the block uses a plain
    labeling prompt over previous/current/next visual context. If a seed
    subtask is available, the block uses the seed-aware relabeling prompt and
    falls back to the seed subtask when the provider blocks the prompt and
    ``on_blocked_prompt`` is ``"seed"``.

    Args:
        provider: Inference provider used to label each fixed segment.
        video_key: Key in ``row.videos`` for the video stream used to render
            previous/current/next segment contact sheets.
        segments_column: Row column containing the fixed input segment
            dictionaries. Each segment must contain ``start_sec`` and
            ``end_sec``. If a segment contains ``subtask``, that value is used
            as the seed for the relabeling prompt. Segments without ``subtask``
            use the plain labeling prompt.
        output_column: Row column written by this block. The output is a list of
            segment dictionaries with the same timing fields and a rewritten
            ``subtask`` value. Keeping this separate from
            ``segments_column`` preserves the original segmentation output for
            inspection or comparison.
        frame_width: Width, in pixels, for rendered segment contact-sheet
            frames.
        max_frames_per_segment: Maximum number of frames sampled for each
            previous, current, and next segment sheet.
        columns: Number of columns in each segment contact sheet.
        quality: JPEG quality for rendered contact sheets, from ``1`` to
            ``100``.
        temperature: Sampling temperature passed to the inference provider.
        on_blocked_prompt: Behavior when the provider blocks a labeling prompt.
            ``"seed"`` writes the seed subtask fallback; ``"raise"``
            propagates the provider error.
        max_concurrent_requests: Maximum provider requests allowed concurrently
            per worker.
    """

    from refiner.inference import generate_text

    if not video_key.strip():
        raise ValueError("video_key must be non-empty")
    if not segments_column.strip():
        raise ValueError("segments_column must be non-empty")
    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if max_frames_per_segment <= 0:
        raise ValueError("max_frames_per_segment must be > 0")
    if on_blocked_prompt not in {"seed", "raise"}:
        raise ValueError("on_blocked_prompt must be 'seed' or 'raise'")

    async def _label_subtasks(
        row: Row,
        generate_text: "GenerateTextFn",
    ) -> MapResult:
        if not isinstance(row, RoboticsRow):
            raise TypeError("subtask_labeling expects RoboticsRow inputs")

        video = _resolve_video(row, video_key)
        segments = _normalize_input_segments(row[segments_column])
        instruction = "; ".join(task for task in row.tasks if task.strip())
        segment_sheets = await _segment_contact_sheets(
            video=video,
            segments=segments,
            frame_width=frame_width,
            max_frames=max_frames_per_segment,
            columns=columns,
            quality=quality,
        )
        blank_sheet = _blank_contact_sheet(
            frame_width=frame_width,
            columns=columns,
            quality=quality,
        )
        labeled_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            seed_label = str(segment["subtask"])
            content = await _subtask_labeling_content(
                segments=segments,
                segment_sheets=segment_sheets,
                blank_sheet=blank_sheet,
                segment_index=index,
                instruction=instruction,
                seed_label=seed_label,
            )
            messages = cast(list[Message], [{"role": "user", "content": content}])
            try:
                response = await generate_text(
                    messages=messages,
                    schema=_SubtaskLabelingResult,
                    provider_options={
                        "google": {"safetySettings": GEMINI_BLOCK_NONE_SAFETY_SETTINGS}
                    },
                    temperature=temperature,
                )
            except RuntimeError as exc:
                block_reason = _blocked_prompt_reason(exc)
                if block_reason is None or on_blocked_prompt == "raise":
                    raise
                logger.warning(
                    "subtask labeling provider blocked episode {} segment {}: {}",
                    row.episode_id,
                    index,
                    block_reason,
                )
                label = seed_label
            else:
                parsed = response.object
                if not isinstance(parsed, _SubtaskLabelingResult):
                    raise TypeError(
                        "subtask_labeling expected a structured response object"
                    )
                label = _normalize_label(parsed.subtask) or seed_label

            labeled = dict(segment)
            labeled["subtask"] = label
            labeled_segments.append(labeled)

        return row.update({output_column: labeled_segments})

    return generate_text(
        fn=_label_subtasks,
        provider=provider,
        max_concurrent_requests=max_concurrent_requests,
    )


async def _subtask_labeling_content(
    *,
    segments: Sequence[Mapping[str, Any]],
    segment_sheets: Sequence[TimestampedContactSheet],
    blank_sheet: TimestampedContactSheet,
    segment_index: int,
    instruction: str,
    seed_label: str,
) -> list[dict[str, Any]]:
    segment = segments[segment_index]
    start_sec = float(segment["start_sec"])
    end_sec = float(segment["end_sec"])
    prompt_template = (
        _SUBTASK_SEED_LABELING_PROMPT_TEMPLATE
        if seed_label
        else _SUBTASK_LABELING_PROMPT_TEMPLATE
    )
    text = prompt_template.format(
        instruction=instruction,
        segment_index=segment_index,
        segment_count=len(segments),
        start_sec=start_sec,
        end_sec=end_sec,
        seed_label=seed_label,
    )
    file_parts: list[dict[str, Any]] = []
    for neighbor_index in (segment_index - 1, segment_index, segment_index + 1):
        if 0 <= neighbor_index < len(segments):
            sheet = segment_sheets[neighbor_index]
        else:
            sheet = blank_sheet
        file_parts.append(
            {"type": "file", "mediaType": sheet.media_type, "data": sheet.data}
        )
    return [{"type": "text", "text": text}, *file_parts]


__all__ = ["subtask_labeling"]
