from __future__ import annotations

import io
import math
import re
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from refiner.pipeline.data.row import Row
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.worker.context import logger

if TYPE_CHECKING:
    from PIL import Image

    from refiner.video import VideoSource


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


def _normalize_input_segments(
    value: Any,
    segment_label_key: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("segments_column must contain a sequence of segment mappings")

    segments = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError("segments_column must contain segment mappings")
        item = cast(Mapping[str, Any], item)
        if "start_sec" not in item or "end_sec" not in item:
            raise ValueError("each segment must contain start_sec and end_sec")
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])
        if end_sec <= start_sec:
            continue
        segment = dict(item)
        segment["start_sec"] = round(max(0.0, start_sec), 3)
        segment["end_sec"] = round(max(0.0, end_sec), 3)
        segments.append(segment)

    sorted_segments = sorted(
        segments,
        key=lambda segment: (segment["start_sec"], segment["end_sec"]),
    )
    _log_on_overlapping_segments(sorted_segments)
    return sorted_segments


def _seed_labels(
    *,
    row: Row,
    segments: Sequence[Mapping[str, Any]],
    labels_column: str | None,
    segment_label_key: str,
) -> list[str]:
    labels: Sequence[Any] | None = None
    if labels_column is not None:
        value = row[labels_column]
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise TypeError("labels_column must contain a sequence of labels")
        labels = value
        if len(labels) != len(segments):
            raise ValueError("labels_column length must match segments_column length")

    seed_labels = []
    for index, segment in enumerate(segments):
        raw_label = (
            labels[index] if labels is not None else segment.get(segment_label_key)
        )
        seed_labels.append(_normalize_label(str(raw_label or "")))
    return seed_labels


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).lower()


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


async def _segment_contact_sheet(
    *,
    video: VideoSource,
    start_sec: float,
    end_sec: float,
    frame_width: int,
    max_frames: int,
    columns: int,
    quality: int,
) -> TimestampedContactSheet:
    if end_sec <= start_sec:
        raise ValueError("segment end_sec must be greater than start_sec")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if columns <= 0:
        raise ValueError("columns must be > 0")

    check_required_dependencies(
        "subtask_labeling",
        ["av", ("PIL", "pillow")],
        dist="video",
    )

    frames: list[tuple[float, Image.Image]] = []
    sample_sec = max((end_sec - start_sec) / max_frames, 1e-6)
    async for timestamp, image in _sample_timestamped_frames(
        video,
        sample_sec=sample_sec,
        frame_width=frame_width,
    ):
        if timestamp + 1e-6 < start_sec:
            continue
        if timestamp - 1e-6 > end_sec:
            break
        frames.append((timestamp, image))

    if not frames:
        async for frame in video.iter_frames():
            timestamp = frame.timestamp_s
            if timestamp is None or timestamp + 1e-6 < start_sec:
                continue
            if timestamp - 1e-6 > end_sec:
                break
            image = frame.frame.to_image().convert("RGB")
            if image.width != frame_width:
                from PIL import Image

                height = max(1, round(image.height * frame_width / image.width))
                image = image.resize(
                    (frame_width, height),
                    resample=Image.Resampling.BILINEAR,
                )
            frames.append((timestamp, _draw_timestamp_badge(image, timestamp)))
            break

    if len(frames) > max_frames:
        frames = [frames[index] for index in _uniform_indexes(len(frames), max_frames)]

    if not frames:
        return _blank_contact_sheet(
            frame_width=frame_width,
            columns=columns,
            quality=quality,
        )

    rows = math.ceil(len(frames) / columns)
    sheets = [
        sheet
        async for sheet in _build_contact_sheets(
            _single_batch(frames),
            rows=rows,
            columns=columns,
            quality=quality,
        )
    ]
    return sheets[0]


def _uniform_indexes(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [round(index * (count - 1) / (limit - 1)) for index in range(limit)]


async def _single_batch(
    frames: Sequence[tuple[float, Image.Image]],
) -> AsyncIterator[Sequence[tuple[float, Image.Image]]]:
    yield frames


def _blank_contact_sheet(
    *,
    frame_width: int,
    columns: int,
    quality: int,
) -> TimestampedContactSheet:
    from PIL import Image

    frame_height = max(2, round(frame_width * 9 / 16))
    width = frame_width * columns
    image = Image.new("RGB", (width, frame_height), color=(245, 245, 245))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return TimestampedContactSheet(
        data=output.getvalue(),
        media_type="image/jpeg",
        index=0,
        timestamps=(0.0,),
        width=width,
        height=frame_height,
        rows=1,
        columns=columns,
    )


async def timestamped_contact_sheets(
    video: VideoSource,
    *,
    sample_sec: float = 0.5,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    quality: int = 84,
) -> AsyncIterator[TimestampedContactSheet]:
    """Yield JPEG contact sheets with visible timestamp badges."""

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
    "timestamped_contact_sheets",
]
