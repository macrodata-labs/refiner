from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.robotics.subtask_annotation.utils import (
    _blocked_prompt_reason,
    _iter_timestamped_contact_sheets,
    _log_on_overlapping_segments,
    _parse_json_object,
    _resolve_video,
)
from refiner.worker.context import logger

if TYPE_CHECKING:
    from refiner.inference.generate_text import GenerateTextFn
    from refiner.video import VideoSource


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
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
    temperature: float = 0.1,
    on_blocked_prompt: Literal["empty", "raise"] = "empty",
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that annotates robotics episode subtasks.

    Args:
        provider: Text-generation provider used to run the vision-language model.
        video_key: Video key to annotate.
        output_column: Row column that receives the predicted subtask segments.
        sample_sec: Seconds between sampled video frames in the contact sheets.
        frame_width: Width, in pixels, for each sampled frame tile.
        frames_per_sheet: Maximum number of sampled frames packed into one sheet.
        columns: Number of columns in each contact-sheet grid.
        quality: JPEG quality for contact-sheet images, from 1 to 100.
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
            sample_sec=sample_sec,
            frame_width=frame_width,
            frames_per_sheet=frames_per_sheet,
            columns=columns,
            quality=quality,
        )
        params: dict[str, Any] = {
            "temperature": temperature,
        }
        messages = cast(list[Message], [{"role": "user", "content": content}])
        try:
            response = await generate_text(
                messages=messages,
                schema=_SubtaskAnnotationResult,
                **params,
            )
        except RuntimeError as exc:
            block_reason = _blocked_prompt_reason(exc)
            if block_reason is None or on_blocked_prompt == "raise":
                raise
            logger.warning(
                "subtask annotation provider blocked episode {}: {}",
                row.episode_id,
                block_reason,
            )
            return row.update({output_column: []})
        parsed = (
            response.object
            if isinstance(response.object, _SubtaskAnnotationResult)
            else _parse_subtask_annotation_result(response.text)
        )
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


def _parse_subtask_annotation_result(text: str) -> _SubtaskAnnotationResult:
    return _SubtaskAnnotationResult.model_validate(_parse_json_object(text))


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

    async for sheet in _iter_timestamped_contact_sheets(
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
        label = segment.subtask.strip() or f"segment {index}"
        normalized.append(
            {
                "start_sec": round(max(0.0, float(segment.start_sec)), 3),
                "end_sec": round(max(0.0, float(segment.end_sec)), 3),
                "subtask": label,
            }
        )
    sorted_segments = sorted(
        normalized,
        key=lambda segment: (segment["start_sec"], segment["end_sec"]),
    )
    _log_on_overlapping_segments(sorted_segments)
    return sorted_segments


__all__ = ["subtask_annotation"]
