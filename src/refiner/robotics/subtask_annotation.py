from __future__ import annotations

import io
import json
import math
import re
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any, Literal, cast

from pydantic import BaseModel

from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.worker.context import logger

if TYPE_CHECKING:
    from PIL import Image

    from refiner.inference.generate_text import GenerateTextFn
    from refiner.video import VideoSource

_DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE = """Reconstruct the sequence of manipulation events in this robot video from timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

How to read the images:
- Each image is a contact sheet with {columns_count} columns and {rows_count} rows.
- Time runs left-to-right within each row, then continues on the next row.
- Each tile has a visible timestamp in its top-left corner, such as 012.50s.
- Use the visible timestamp printed inside the tile, not the tile index, when choosing start_sec and end_sec.
- Boundaries should normally land on or near one of the visible timestamps.

Rules:
- Treat each segment as one event that changes what is true about the world.
- Good event boundaries happen when an object becomes held, is released, reaches a new location, a lid/door changes open/closed state, a tool starts/stops affecting a surface, or contents visibly move.
- For each event, choose start_sec at the first timestamp where the causal motion for that event is underway, and end_sec at the first timestamp where the resulting world state is achieved.
- If an action is continuous and changes the same state gradually, keep it as one event.
- If the same action repeats on different objects or target locations, output separate repeated events.
- Avoid segments for idle time, camera motion, hesitation, or tiny hand adjustments.
"""


class _SubtaskSegment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str


class _SubtaskAnnotationResult(BaseModel):
    segments: list[_SubtaskSegment]


@dataclass(frozen=True, slots=True)
class TimestampedContactSheet:
    data: bytes
    media_type: str
    index: int
    timestamps: tuple[float, ...]
    width: int
    height: int
    rows: int
    columns: int

    @property
    def start_sec(self) -> float:
        return self.timestamps[0]

    @property
    def end_sec(self) -> float:
        return self.timestamps[-1]

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)


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
    min_segment_duration_sec: float | None = 0.0,
    include_contact_sheet_manifest: bool = False,
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
        min_segment_duration_sec: Minimum segment duration to keep. Set to
            ``None`` to disable duration filtering.
        include_contact_sheet_manifest: Include textual sheet ranges in the prompt.
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
    if min_segment_duration_sec is not None and min_segment_duration_sec < 0:
        raise ValueError("min_segment_duration_sec must be >= 0")
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
            include_contact_sheet_manifest=include_contact_sheet_manifest,
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
        if min_segment_duration_sec is not None:
            segments = [
                segment
                for segment in segments
                if float(segment["end_sec"]) - float(segment["start_sec"])
                >= min_segment_duration_sec
            ]
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


def _blocked_prompt_reason(exc: RuntimeError) -> str | None:
    marker = "promptFeedback.blockReason="
    _, found, reason = str(exc).rpartition(marker)
    if not found:
        return None
    reason = reason.strip()
    return reason or None


def contact_sheet_prompt_manifest(
    sheets: Iterable[TimestampedContactSheet],
) -> str:
    """Describe ordered contact sheets for a multimodal task prompt."""

    lines = [
        "The following contact sheets are ordered chronologically.",
        "Each tile is a sampled video frame with its timestamp burned into the "
        "top-left corner.",
        "Actions may continue across contact sheet boundaries; do not create a "
        "segment boundary just because the next image is a new sheet.",
    ]
    seen = False
    for sheet in sheets:
        seen = True
        lines.append(
            f"Sheet {sheet.index}: {sheet.frame_count} frames, "
            f"{sheet.rows}x{sheet.columns} grid, "
            f"{sheet.start_sec:.2f}s through {sheet.end_sec:.2f}s."
        )
    if not seen:
        raise ValueError("sheets must be non-empty")
    return "\n".join(lines)


def _resolve_video(row: RoboticsRow, video_key: str) -> VideoSource:
    if not row.videos:
        raise ValueError(f"episode {row.episode_id!r} has no videos")
    if video_key not in row.videos:
        raise ValueError(
            f"episode {row.episode_id!r} is missing video key {video_key!r}"
        )
    return row.videos[video_key]


def _parse_subtask_annotation_result(text: str) -> _SubtaskAnnotationResult:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])

    return _SubtaskAnnotationResult.model_validate(value)


async def _subtask_annotation_content(
    *,
    video: VideoSource,
    tasks: list[str],
    sample_sec: float,
    frame_width: int,
    frames_per_sheet: int,
    columns: int,
    quality: int,
    include_contact_sheet_manifest: bool,
) -> list[dict[str, Any]]:
    sheets_for_manifest: list[TimestampedContactSheet] = []
    file_parts: list[dict[str, Any]] = []
    first_sheet: TimestampedContactSheet | None = None

    async for sheet in _iter_timestamped_contact_sheets(
        video,
        sample_sec=sample_sec,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        quality=quality,
    ):
        if first_sheet is None:
            first_sheet = sheet
        if include_contact_sheet_manifest:
            sheets_for_manifest.append(sheet)
        file_parts.append(
            {"type": "file", "mediaType": sheet.media_type, "data": sheet.data}
        )

    if first_sheet is None:
        raise ValueError("video produced no frames")

    text = _DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE.replace(
        "{columns_count}",
        str(first_sheet.columns),
    ).replace("{rows_count}", str(first_sheet.rows))
    instruction = "; ".join(task for task in tasks if task.strip())
    if instruction:
        text = f"{text}\nEpisode instruction: {instruction}\n"
    if include_contact_sheet_manifest:
        text = f"{text}\n\n{contact_sheet_prompt_manifest(sheets_for_manifest)}"
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


def _log_on_overlapping_segments(segments: Sequence[Mapping[str, Any]]) -> None:
    previous: Mapping[str, Any] | None = None
    for segment in segments:
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        if previous is not None and start_sec < float(previous["end_sec"]):
            previous_start_sec = float(previous["start_sec"])
            previous_end_sec = float(previous["end_sec"])
            logger.warning(
                "subtask annotation produced overlapping segments: "
                "{:.3f}s-{:.3f}s overlaps {:.3f}s-{:.3f}s",
                previous_start_sec,
                previous_end_sec,
                start_sec,
                end_sec,
            )
        previous = segment


async def timestamped_contact_sheets(
    video: VideoSource,
    *,
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
) -> list[TimestampedContactSheet]:
    """Sample a video into JPEG contact sheets with visible timestamp badges."""

    return [
        sheet
        async for sheet in _iter_timestamped_contact_sheets(
            video,
            sample_sec=sample_sec,
            frame_width=frame_width,
            frames_per_sheet=frames_per_sheet,
            columns=columns,
            quality=quality,
        )
    ]


async def _iter_timestamped_contact_sheets(
    video: VideoSource,
    *,
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
) -> AsyncIterator[TimestampedContactSheet]:
    """Yield JPEG contact sheets without retaining every sampled frame."""

    if sample_sec <= 0:
        raise ValueError("sample_sec must be > 0")
    if frame_width <= 0:
        raise ValueError("frame_width must be > 0")
    if frames_per_sheet <= 0:
        raise ValueError("frames_per_sheet must be > 0")
    if columns <= 0:
        raise ValueError("columns must be > 0")
    if quality <= 0 or quality > 100:
        raise ValueError("quality must be between 1 and 100")

    check_required_dependencies(
        "timestamped_contact_sheets",
        ["av", ("PIL", "pillow")],
        dist="video",
    )

    frame_batches = _batched(
        _sample_timestamped_frames(
            video,
            sample_sec=sample_sec,
            frame_width=frame_width,
        ),
        frames_per_sheet,
    )
    produced = False
    async for sheet in _build_contact_sheets(
        frame_batches,
        rows=math.ceil(frames_per_sheet / columns),
        columns=columns,
        quality=quality,
    ):
        produced = True
        yield sheet

    if not produced:
        raise ValueError("video produced no frames")


async def _sample_timestamped_frames(
    video: VideoSource,
    *,
    sample_sec: float,
    frame_width: int,
) -> AsyncIterator[tuple[float, Image.Image]]:
    next_timestamp = 0.0

    async for frame in video.iter_frames():
        timestamp = frame.timestamp_s
        if timestamp is None or timestamp + 1e-6 < next_timestamp:
            continue

        image = frame.frame.to_image().convert("RGB")
        if image.width != frame_width:
            from PIL import Image

            height = max(1, round(image.height * frame_width / image.width))
            image = image.resize(
                (frame_width, height),
                resample=Image.Resampling.BILINEAR,
            )
        yield timestamp, _draw_timestamp_badge(image, timestamp)
        next_timestamp = timestamp + sample_sec


async def _batched(
    frames: AsyncIterator[tuple[float, Image.Image]],
    size: int,
) -> AsyncIterator[Sequence[tuple[float, Image.Image]]]:
    batch: list[tuple[float, Image.Image]] = []
    async for frame in frames:
        batch.append(frame)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _build_contact_sheets(
    batches: AsyncIterator[Sequence[tuple[float, Image.Image]]],
    *,
    rows: int,
    columns: int,
    quality: int,
) -> AsyncIterator[TimestampedContactSheet]:
    from PIL import Image

    async for sheet_index, chunk in _aenumerate(batches, start=1):
        frame_width, frame_height = chunk[0][1].size
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows
        sheet = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

        for index, (_, image) in enumerate(chunk):
            x = (index % columns) * frame_width
            y = (index // columns) * frame_height
            sheet.paste(image, (x, y))

        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=quality)
        yield TimestampedContactSheet(
            data=output.getvalue(),
            media_type="image/jpeg",
            index=sheet_index,
            timestamps=tuple(timestamp for timestamp, _ in chunk),
            width=sheet_width,
            height=sheet_height,
            rows=rows,
            columns=columns,
        )


async def _aenumerate(
    values: AsyncIterator[Sequence[tuple[float, Image.Image]]],
    *,
    start: int,
) -> AsyncIterator[tuple[int, Sequence[tuple[float, Image.Image]]]]:
    index = start
    async for value in values:
        yield index, value
        index += 1


def _draw_timestamp_badge(image: Image.Image, timestamp: float) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    label = f"{timestamp:06.2f}s"

    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    padding = max(4, round(min(image.width, image.height) * 0.018))
    badge_width = text_width + padding * 2
    badge_height = text_height + padding * 2

    draw.rectangle((0, 0, badge_width, badge_height), fill=(0, 0, 0))
    draw.text(
        (padding - left, padding - top),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    return result


__all__ = [
    "TimestampedContactSheet",
    "contact_sheet_prompt_manifest",
    "subtask_annotation",
    "timestamped_contact_sheets",
]
