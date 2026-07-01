from __future__ import annotations

import io
import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from refiner.inference.types import GoogleSafetySetting
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.worker.context import logger

if TYPE_CHECKING:
    from PIL import Image, ImageFont

    from refiner.video import VideoSource


GEMINI_BLOCK_NONE_SAFETY_SETTINGS: list[GoogleSafetySetting] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


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
        segments.append(
            {
                "start_sec": round(max(0.0, start_sec), 3),
                "end_sec": round(max(0.0, end_sec), 3),
                "subtask": _normalize_label(str(item.get("subtask") or "")),
            }
        )

    sorted_segments = sorted(
        segments,
        key=lambda segment: (segment["start_sec"], segment["end_sec"]),
    )
    _log_on_overlapping_segments(sorted_segments)
    return sorted_segments


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
                    resample=Image.Resampling.BOX,
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

    return _build_contact_sheet(
        frames,
        sheet_index=1,
        rows=math.ceil(len(frames) / columns),
        columns=columns,
        quality=quality,
    )


async def _segment_contact_sheets(
    *,
    video: VideoSource,
    segments: Sequence[Mapping[str, Any]],
    frame_width: int,
    max_frames: int,
    columns: int,
    quality: int,
) -> list[TimestampedContactSheet]:
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if columns <= 0:
        raise ValueError("columns must be > 0")

    check_required_dependencies(
        "subtask_labeling",
        ["av", ("PIL", "pillow")],
        dist="video",
    )

    targets: list[tuple[float, int]] = []
    for segment_index, segment in enumerate(segments):
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        if end_sec <= start_sec:
            continue
        for timestamp in _segment_target_timestamps(
            start_sec,
            end_sec,
            max_frames,
        ):
            targets.append((timestamp, segment_index))

    if not targets:
        return []

    targets.sort(key=lambda item: item[0])
    frames_by_segment: list[list[tuple[float, Image.Image]]] = [
        [] for _ in range(len(segments))
    ]
    target_index = 0
    last_image: Image.Image | None = None

    async for frame in video.iter_frames():
        timestamp = frame.timestamp_s
        if timestamp is None:
            continue
        if target_index >= len(targets):
            break
        if timestamp + 1e-6 < targets[target_index][0]:
            continue

        image = frame.frame.to_image().convert("RGB")
        if image.width != frame_width:
            from PIL import Image

            height = max(1, round(image.height * frame_width / image.width))
            image = image.resize(
                (frame_width, height),
                resample=Image.Resampling.BOX,
            )
        last_image = image

        while (
            target_index < len(targets) and timestamp + 1e-6 >= targets[target_index][0]
        ):
            target_timestamp, segment_index = targets[target_index]
            frames_by_segment[segment_index].append(
                (
                    target_timestamp,
                    _draw_timestamp_badge(image, target_timestamp),
                )
            )
            target_index += 1

    if last_image is not None:
        while target_index < len(targets):
            target_timestamp, segment_index = targets[target_index]
            frames_by_segment[segment_index].append(
                (
                    target_timestamp,
                    _draw_timestamp_badge(last_image, target_timestamp),
                )
            )
            target_index += 1

    blank_sheet = _blank_contact_sheet(
        frame_width=frame_width,
        columns=columns,
        quality=quality,
    )
    sheets: list[TimestampedContactSheet] = []
    for frames in frames_by_segment:
        if not frames:
            sheets.append(blank_sheet)
            continue
        sheets.append(
            _build_contact_sheet(
                frames,
                sheet_index=1,
                rows=math.ceil(len(frames) / columns),
                columns=columns,
                quality=quality,
            )
        )
    return sheets


def _segment_target_timestamps(
    start_sec: float,
    end_sec: float,
    max_frames: int,
) -> list[float]:
    if max_frames == 1:
        return [round((start_sec + end_sec) / 2, 6)]
    return [
        round(start_sec + index * (end_sec - start_sec) / (max_frames - 1), 6)
        for index in range(max_frames)
    ]


def _uniform_indexes(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [round(index * (count - 1) / (limit - 1)) for index in range(limit)]


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
    image.save(output, format="JPEG", quality=quality, subsampling=2)
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
    quality: int = 95,
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
    target_timestamp = 0.0

    async for frame in video.iter_frames():
        timestamp = frame.timestamp_s
        if timestamp is None or timestamp + 1e-6 < target_timestamp:
            continue

        image = frame.frame.to_image().convert("RGB")
        if image.width != frame_width:
            from PIL import Image

            height = max(1, round(image.height * frame_width / image.width))
            image = image.resize(
                (frame_width, height),
                resample=Image.Resampling.BOX,
            )
        while timestamp + 1e-6 >= target_timestamp:
            sampled_timestamp = round(target_timestamp, 6)
            yield sampled_timestamp, _draw_timestamp_badge(image, sampled_timestamp)
            target_timestamp += sample_sec


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
    sheet_index = 1
    async for chunk in batches:
        yield _build_contact_sheet(
            chunk,
            sheet_index=sheet_index,
            rows=rows,
            columns=columns,
            quality=quality,
        )
        sheet_index += 1


def _build_contact_sheet(
    frames: Sequence[tuple[float, Image.Image]],
    *,
    sheet_index: int,
    rows: int,
    columns: int,
    quality: int,
) -> TimestampedContactSheet:
    from PIL import Image

    frame_width, frame_height = frames[0][1].size
    sheet_width = frame_width * columns
    sheet_height = frame_height * rows
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(0, 0, 0))

    for index, (_, image) in enumerate(frames):
        x = (index % columns) * frame_width
        y = (index // columns) * frame_height
        sheet.paste(image, (x, y))

    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=quality, subsampling=2)
    return TimestampedContactSheet(
        data=output.getvalue(),
        media_type="image/jpeg",
        index=sheet_index,
        timestamps=tuple(timestamp for timestamp, _ in frames),
        width=sheet_width,
        height=sheet_height,
        rows=rows,
        columns=columns,
    )


def _draw_timestamp_badge(image: Image.Image, timestamp: float) -> Image.Image:
    from PIL import ImageDraw

    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = _load_timestamp_font()
    label = f"{timestamp:06.2f}s"

    draw.rectangle((0, 0, 72, 26), fill=(0, 0, 0))
    draw.text((7, 3), label, fill=(255, 255, 255), font=font)
    return result


def _load_timestamp_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, 17)
        except OSError:
            pass
    return ImageFont.load_default()


__all__ = [
    "TimestampedContactSheet",
    "timestamped_contact_sheets",
]
