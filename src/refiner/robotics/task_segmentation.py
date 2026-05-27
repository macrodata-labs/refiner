from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from refiner.utils import check_required_dependencies

if TYPE_CHECKING:
    from PIL import Image

    from refiner.video import VideoFile


@dataclass(frozen=True, slots=True)
class TimestampedContactSheet:
    data: bytes
    media_type: str
    timestamps: tuple[float, ...]
    width: int
    height: int


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
    for start in range(0, len(samples), frames_per_sheet):
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
                timestamps=tuple(timestamp for timestamp, _ in chunk),
                width=sheet_width,
                height=sheet_height,
            )
        )

    return sheets


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


__all__ = [
    "TimestampedContactSheet",
    "timestamped_contact_sheets",
]
