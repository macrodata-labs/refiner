from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any, TypeAlias, cast

from pydantic import BaseModel

from refiner.inference.types import Message, ProviderOptions
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import MapResult
from refiner.robotics.lerobot_format import LeRobotRow
from refiner.utils import check_required_dependencies

if TYPE_CHECKING:
    from PIL import Image

    from refiner.inference.generate_text import GenerateTextFn
    from refiner.inference.providers import (
        AnthropicEndpointProvider,
        GoogleEndpointProvider,
        OpenAIEndpointProvider,
        OpenAIResponsesProvider,
        VLLMProvider,
    )
    from refiner.video import VideoFile

_DEFAULT_SUBTASK_ANNOTATION_PROMPT = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Treat each segment as one event that changes what is true about the world.
- Good event boundaries happen when an object becomes held, is released, reaches a new location, a lid/door changes open/closed state, a tool starts/stops affecting a surface, or contents visibly move.
- For each event, choose start_sec at the first timestamp where the causal motion for that event is underway, and end_sec at the first timestamp where the resulting world state is achieved.
- If an action is continuous and changes the same state gradually, keep it as one event.
- If the same action repeats on different objects or target locations, output separate repeated events.
- Avoid idle time, camera motion, hesitation, and tiny hand adjustments.
"""

if TYPE_CHECKING:
    SubtaskAnnotationProvider: TypeAlias = (
        AnthropicEndpointProvider
        | GoogleEndpointProvider
        | OpenAIEndpointProvider
        | OpenAIResponsesProvider
        | VLLMProvider
    )
else:
    SubtaskAnnotationProvider: TypeAlias = Any


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
    provider: SubtaskAnnotationProvider,
    video_key: str | None = None,
    prompt: str = _DEFAULT_SUBTASK_ANNOTATION_PROMPT,
    output_column: str = "predicted_subtasks",
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
    temperature: float = 0.1,
    min_segment_duration_sec: float | None = 3.5,
    include_contact_sheet_manifest: bool = False,
    providerOptions: ProviderOptions | None = None,
    generation_params: Mapping[str, Any] | None = None,
    max_concurrent_requests: int = 256,
) -> Callable[[Row], Any]:
    """Return an async map block that annotates LeRobot episode subtasks."""

    from refiner.inference import generate_text

    if not output_column.strip():
        raise ValueError("output_column must be non-empty")
    if min_segment_duration_sec is not None and min_segment_duration_sec < 0:
        raise ValueError("min_segment_duration_sec must be >= 0")

    async def _annotate_subtasks(
        row: Row,
        generate_text: "GenerateTextFn",
    ) -> MapResult:
        if not isinstance(row, LeRobotRow):
            raise TypeError("subtask_annotation expects rows from read_lerobot(...)")

        selected_video_key = _resolve_video_key(row, video_key)
        video = row.videos[selected_video_key]
        content = await _subtask_annotation_content(
            video=video,
            prompt=_prompt_with_instruction(prompt, row.tasks),
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
            providerOptions=providerOptions,
            schema=_SubtaskAnnotationResult,
            **params,
        )
        parsed = (
            response.object
            if isinstance(response.object, _SubtaskAnnotationResult)
            else _parse_subtask_annotation_result(response.text)
        )
        segments = _filter_segments(
            _normalize_segments(parsed.segments),
            min_duration_sec=min_segment_duration_sec,
        )
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


def _resolve_video_key(row: LeRobotRow, video_key: str | None) -> str:
    video_keys = list(row.videos)
    if video_key is not None:
        if video_key not in video_keys:
            raise ValueError(
                f"episode {row.episode_index} is missing video key {video_key!r}"
            )
        return video_key
    if not video_keys:
        raise ValueError(f"episode {row.episode_index} has no videos")
    return video_keys[0]


def _prompt_with_instruction(prompt: str, tasks: list[str]) -> str:
    instruction = "; ".join(task for task in tasks if task.strip())
    if not instruction:
        return prompt
    return f"{prompt}\nEpisode instruction: {instruction}\n"


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
    video: VideoFile,
    prompt: str,
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
    text = prompt
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


def _filter_segments(
    segments: list[dict[str, Any]],
    *,
    min_duration_sec: float | None,
) -> list[dict[str, Any]]:
    if min_duration_sec is None:
        return segments
    return [
        segment
        for segment in segments
        if float(segment["end_sec"]) - float(segment["start_sec"]) >= min_duration_sec
    ]


async def timestamped_contact_sheets(
    video: VideoFile,
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
    video: VideoFile,
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
        image = _resize_to_width(image, frame_width)
        image = _draw_timestamp_badge(image, timestamp)
        samples.append((timestamp, image))
        next_timestamp = timestamp + sample_sec

    return samples


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image

    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height))


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

        sheets.append(
            TimestampedContactSheet(
                data=_encode_jpeg(sheet, quality=quality),
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


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


__all__ = [
    "SubtaskAnnotationProvider",
    "TimestampedContactSheet",
    "contact_sheet_prompt_manifest",
    "subtask_annotation",
    "timestamped_contact_sheets",
]
