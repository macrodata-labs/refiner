from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from refiner.inference import InferenceSchemaValidationError
from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.robotics.subtask_annotation.utils import (
    GEMINI_BLOCK_NONE_SAFETY_SETTINGS,
    _blocked_prompt_reason,
    _log_on_overlapping_segments,
    _resolve_video,
    timestamped_contact_sheets,
)
from refiner.worker.context import logger

if TYPE_CHECKING:
    from refiner.inference.generate_text import GenerateTextFn
    from refiner.video import VideoSource


_SEGMENTATION_CONTACT_SHEET_QUALITY = 95
_SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY = 70
_SEGMENTATION_SAMPLE_SEC = 0.5
_SEGMENTATION_FRAME_WIDTH = 224
_SEGMENTATION_FRAMES_PER_SHEET = 20
_SEGMENTATION_COLUMNS = 5
_STRUCTURED_OUTPUT_ATTEMPTS = 3

_DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
"""


class _SubtaskSegment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str


class _SubtaskAnnotationResult(BaseModel):
    segments: list[_SubtaskSegment]


def subtask_annotation(
    *,
    provider: InferenceProvider = GoogleEndpointProvider(model="gemini-3.5-flash"),
    video_key: str,
    output_column: str = "predicted_subtasks",
    temperature: float = 0.1,
    on_blocked_prompt: Literal["empty", "raise"] = "empty",
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that annotates robotics episode subtasks.

    Args:
        provider: Text-generation provider used to run the vision-language model.
        video_key: Video key to annotate.
        output_column: Row column that receives the predicted subtask segments.
        temperature: Generation temperature passed to the provider request.
        on_blocked_prompt: Behavior when the provider blocks the prompt before
            returning candidates. ``"empty"`` writes an empty segment list and
            logs the block; ``"raise"`` propagates the provider error.
        max_concurrent_requests: Maximum provider requests allowed at once per
            worker.
    """

    from refiner.inference import generate_text

    if not video_key.strip():
        raise ValueError("video_key must be non-empty")
    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if on_blocked_prompt not in {"empty", "raise"}:
        raise ValueError("on_blocked_prompt must be 'empty' or 'raise'")

    async def _annotate_subtasks(
        row: Row,
        generate_text: "GenerateTextFn",
    ) -> MapResult:
        if not isinstance(row, RoboticsRow):
            raise TypeError("subtask_annotation expects RoboticsRow inputs")

        video = _resolve_video(row, video_key)
        content = await _subtask_annotation_content(
            video=video,
            tasks=row.tasks,
            sample_sec=_SEGMENTATION_SAMPLE_SEC,
            frame_width=_SEGMENTATION_FRAME_WIDTH,
            frames_per_sheet=_SEGMENTATION_FRAMES_PER_SHEET,
            columns=_SEGMENTATION_COLUMNS,
            quality=_SEGMENTATION_CONTACT_SHEET_QUALITY,
        )
        try:
            response = await _request_subtask_annotation(
                generate_text=generate_text,
                content=content,
                temperature=temperature,
            )
        except RuntimeError as exc:
            block_reason = _blocked_prompt_reason(exc)
            if block_reason is None:
                raise
            try:
                fallback_content = await _subtask_annotation_content(
                    video=video,
                    tasks=row.tasks,
                    sample_sec=_SEGMENTATION_SAMPLE_SEC,
                    frame_width=_SEGMENTATION_FRAME_WIDTH,
                    frames_per_sheet=_SEGMENTATION_FRAMES_PER_SHEET,
                    columns=_SEGMENTATION_COLUMNS,
                    quality=_SEGMENTATION_FALLBACK_CONTACT_SHEET_QUALITY,
                )
                response = await _request_subtask_annotation(
                    generate_text=generate_text,
                    content=fallback_content,
                    temperature=temperature,
                )
            except RuntimeError as fallback_exc:
                fallback_block_reason = _blocked_prompt_reason(fallback_exc)
                if fallback_block_reason is None or on_blocked_prompt == "raise":
                    raise
                block_reason = fallback_block_reason
            else:
                parsed = response.object
                if not isinstance(parsed, _SubtaskAnnotationResult):
                    raise TypeError(
                        "subtask_annotation expected a structured response object"
                    )
                segments = _normalize_segments(parsed.segments)
                return row.update(
                    {
                        output_column: segments,
                    }
                )
            logger.warning(
                "subtask annotation provider blocked episode {}: {}",
                row.episode_id,
                block_reason,
            )
            return row.update({output_column: []})
        parsed = response.object
        if not isinstance(parsed, _SubtaskAnnotationResult):
            raise TypeError("subtask_annotation expected a structured response object")
        segments = _normalize_segments(parsed.segments)
        return row.update(
            {
                output_column: segments,
            }
        )

    return generate_text(
        fn=_annotate_subtasks,
        provider=provider,
        max_concurrent_requests=max_concurrent_requests,
    )


async def _request_subtask_annotation(
    *,
    generate_text: "GenerateTextFn",
    content: list[dict[str, Any]],
    temperature: float,
) -> Any:
    messages = cast(list[Message], [{"role": "user", "content": content}])
    for attempt in range(1, _STRUCTURED_OUTPUT_ATTEMPTS + 1):
        try:
            return await generate_text(
                messages=messages,
                schema=_SubtaskAnnotationResult,
                provider_options={
                    "google": {
                        "safetySettings": GEMINI_BLOCK_NONE_SAFETY_SETTINGS,
                    }
                },
                maxRetries=4,
                temperature=temperature,
            )
        except InferenceSchemaValidationError:
            if attempt == _STRUCTURED_OUTPUT_ATTEMPTS:
                raise
            logger.warning(
                "subtask annotation structured output validation failed; "
                "retrying request attempt={}/{}",
                attempt,
                _STRUCTURED_OUTPUT_ATTEMPTS,
            )
    raise AssertionError("structured output retry loop exited unexpectedly")


async def _subtask_annotation_content(
    *,
    video: VideoSource,
    tasks: list[str],
    sample_sec: float,
    frame_width: int,
    frames_per_sheet: int,
    columns: int,
    quality: int,
) -> list[dict[str, Any]]:
    file_parts: list[dict[str, Any]] = []

    async for sheet in timestamped_contact_sheets(
        video,
        sample_sec=sample_sec,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        quality=quality,
    ):
        file_parts.append(
            {"type": "file", "mediaType": sheet.media_type, "data": sheet.data}
        )

    if not file_parts:
        raise ValueError("video produced no frames")

    text = _DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE
    instruction = "; ".join(task for task in tasks if task.strip())
    if instruction:
        text = f"{text}\nEpisode instruction: {instruction}\n"
    return [
        {
            "type": "text",
            "text": text,
        },
        *file_parts,
    ]


def _normalize_segments(segments: list[_SubtaskSegment]) -> list[dict[str, Any]]:
    normalized = []
    for index, segment in enumerate(segments):
        if segment.end_sec <= segment.start_sec:
            continue
        subtask = segment.subtask.strip() or f"segment {index}"
        normalized.append(
            {
                "start_sec": round(max(0.0, float(segment.start_sec)), 3),
                "end_sec": round(max(0.0, float(segment.end_sec)), 3),
                "subtask": subtask,
            }
        )
    sorted_segments = sorted(
        normalized,
        key=lambda segment: (segment["start_sec"], segment["end_sec"]),
    )
    _log_on_overlapping_segments(sorted_segments)
    return sorted_segments


__all__ = ["subtask_annotation"]
