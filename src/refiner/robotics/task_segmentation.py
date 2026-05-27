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
    "TimestampedContactSheet",
    "contact_sheet_prompt_manifest",
    "timestamped_contact_sheets",
]
