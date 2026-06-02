from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any, cast

from pydantic import BaseModel

from refiner.inference.providers import GoogleEndpointProvider
from refiner.inference.types import InferenceProvider, Message, ProviderOptions
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies

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
    video_key: str | None = None,
    output_column: str = "predicted_subtasks",
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
    temperature: float = 0.1,
    min_segment_duration_sec: float | None = 0.0,
    include_contact_sheet_manifest: bool = False,
    provider_options: ProviderOptions | None = None,
    generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that annotates robotics episode subtasks.

    Args:
        provider: Text-generation provider used to run the vision-language model.
        video_key: Video key to annotate. When omitted, the first episode video is
            used.
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
        provider_options: Provider-specific options passed to the generation request.
        generation_params: Additional generation parameters merged into the request.
        max_concurrent_requests: Maximum provider requests allowed at once per
            worker.
    """

    from refiner.inference import generate_text

    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if min_segment_duration_sec is not None and min_segment_duration_sec < 0:
        raise ValueError("min_segment_duration_sec must be >= 0")

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
            **dict(generation_params or {}),
        }
        messages = cast(list[Message], [{"role": "user", "content": content}])
        response = await generate_text(
            messages=messages,
            provider_options=provider_options,
            schema=_SubtaskAnnotationResult,
            **params,
        )
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


def contact_sheet_prompt_manifest(
    sheets: list[TimestampedContactSheet],
) -> str:
    """Describe ordered contact sheets for a multimodal task prompt."""

    if not sheets:
        raise ValueError("sheets must be non-empty")

    lines = [
        "The following contact sheets are ordered chronologically.",
        "Each tile is a sampled video frame with its timestamp burned into the "
        "top-left corner.",
        "Actions may continue across contact sheet boundaries; do not create a "
        "segment boundary just because the next image is a new sheet.",
    ]
    for sheet in sheets:
        lines.append(
            f"Sheet {sheet.index}: {sheet.frame_count} frames, "
            f"{sheet.rows}x{sheet.columns} grid, "
            f"{sheet.start_sec:.2f}s through {sheet.end_sec:.2f}s."
        )
    return "\n".join(lines)


def _resolve_video(row: RoboticsRow, video_key: str | None) -> VideoSource:
    if video_key is not None:
        if video_key not in row.videos:
            raise ValueError(
                f"episode {row.episode_id!r} is missing video key {video_key!r}"
            )
        return row.videos[video_key]
    for video in row.videos.values():
        return video
    else:
        raise ValueError(f"episode {row.episode_id!r} has no videos")


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
    sheets = await timestamped_contact_sheets(
        video,
        sample_sec=sample_sec,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        quality=quality,
    )
    first_sheet = sheets[0]
    text = _DEFAULT_SUBTASK_ANNOTATION_PROMPT_TEMPLATE.replace(
        "{columns_count}",
        str(first_sheet.columns),
    ).replace("{rows_count}", str(first_sheet.rows))
    instruction = "; ".join(task for task in tasks if task.strip())
    if instruction:
        text = f"{text}\nEpisode instruction: {instruction}\n"
    if include_contact_sheet_manifest:
        text = f"{text}\n\n{contact_sheet_prompt_manifest(sheets)}"
    return [
        {
            "type": "text",
            "text": text,
        },
        *[
            {"type": "file", "mediaType": sheet.media_type, "data": sheet.data}
            for sheet in sheets
        ],
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
    return sorted(
        normalized,
        key=lambda segment: (segment["start_sec"], segment["end_sec"]),
    )


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

    samples = await _sample_timestamped_frames(
        video,
        sample_sec=sample_sec,
        frame_width=frame_width,
    )
    if not samples:
        raise ValueError("video produced no frames")

    return _build_contact_sheets(
        samples,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        quality=quality,
    )


async def _sample_timestamped_frames(
    video: VideoSource,
    *,
    sample_sec: float,
    frame_width: int,
) -> list[tuple[float, Image.Image]]:
    samples: list[tuple[float, Image.Image]] = []
    next_timestamp = 0.0

    async for frame in video.iter_frames():
        timestamp = frame.timestamp_s
        if timestamp is None or timestamp + 1e-6 < next_timestamp:
            continue

        image = frame.frame.to_image().convert("RGB")
        if image.width != frame_width:
            # resize
            height = max(1, round(image.height * frame_width / image.width))
            image = image.resize((frame_width, height))
        image = _draw_timestamp_badge(image, timestamp)
        samples.append((timestamp, image))
        next_timestamp = timestamp + sample_sec

    return samples


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


def _build_contact_sheets(
    samples: list[tuple[float, Image.Image]],
    *,
    frames_per_sheet: int,
    columns: int,
    quality: int,
) -> list[TimestampedContactSheet]:
    from PIL import Image

    frame_width, frame_height = samples[0][1].size
    rows = math.ceil(frames_per_sheet / columns)
    sheet_width = frame_width * columns
    sheet_height = frame_height * rows

    sheets: list[TimestampedContactSheet] = []
    for sheet_index, start in enumerate(range(0, len(samples), frames_per_sheet), 1):
        chunk = samples[start : start + frames_per_sheet]
        sheet = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

        for index, (_, image) in enumerate(chunk):
            x = (index % columns) * frame_width
            y = (index // columns) * frame_height
            sheet.paste(image, (x, y))

        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=quality)
        sheets.append(
            TimestampedContactSheet(
                data=output.getvalue(),
                media_type="image/jpeg",
                index=sheet_index,
                timestamps=tuple(timestamp for timestamp, _ in chunk),
                width=sheet_width,
                height=sheet_height,
                rows=rows,
                columns=columns,
            )
        )

    return sheets


__all__ = [
    "TimestampedContactSheet",
    "contact_sheet_prompt_manifest",
    "subtask_annotation",
    "timestamped_contact_sheets",
]
