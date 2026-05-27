from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

import refiner as mdr
from refiner.io import DataFile


def _write_video(path, *, num_frames: int = 6, fps: int = 5) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"

        for value in range(num_frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((12, 16, 3), 64 + value * 8, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)


async def _build_sheets(video: mdr.video.VideoFile):
    return await mdr.robotics.timestamped_contact_sheets(
        video,
        sample_sec=0.4,
        frame_width=64,
        frames_per_sheet=2,
        columns=2,
        quality=95,
    )


def test_timestamped_contact_sheets_sample_and_tile_video(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=6, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheets = asyncio.run(_build_sheets(video))

    assert len(sheets) == 2
    assert sheets[0].media_type == "image/jpeg"
    assert sheets[0].timestamps == (0.0, 0.4)
    assert sheets[1].timestamps == (0.8,)
    assert sheets[0].width == 128
    assert sheets[0].height == 48
    assert sheets[0].data.startswith(b"\xff\xd8")


def test_timestamped_contact_sheets_engravings_are_visible(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "video.mp4"
    _write_video(path, num_frames=2, fps=5)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    sheet = asyncio.run(_build_sheets(video))[0]
    image = Image.open(io.BytesIO(sheet.data)).convert("RGB")
    pixels = np.asarray(image)

    badge = pixels[:14, :48]
    plain_frame_area = pixels[18:34, :48]

    assert badge.min() < 16
    assert badge.max() > 220
    assert plain_frame_area.mean() > badge.mean()


def test_timestamped_contact_sheets_reject_invalid_options(tmp_path) -> None:
    path = tmp_path / "video.mp4"
    _write_video(path)
    video = mdr.video.VideoFile(DataFile.resolve(path))

    with pytest.raises(ValueError, match="sample_sec must be > 0"):
        asyncio.run(mdr.robotics.timestamped_contact_sheets(video, sample_sec=0))
