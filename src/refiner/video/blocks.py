from __future__ import annotations

from dataclasses import asdict
import math
import uuid

from refiner.io import DataFolder
from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.builtins import describe_builtin
from refiner.pipeline.data.row import Row
from refiner.pipeline.steps import AsyncMapFn
from refiner.video.nvenc import (
    NVENCConfig,
    _local_path,
    encode_image_sequence,
    transcode_video,
)


def transcode_videos(
    *,
    video_key: str,
    output_folder: DataFolderLike,
    output_key: str = "transcoded_video",
    config: NVENCConfig | None = None,
) -> AsyncMapFn:
    """Build an NVENC ``map_async`` block; start with max_in_flight=4 on one L4.

    Rows contain whole VideoFile objects, DataFile objects, or paths. Each result
    adds a local MP4 path. The calling job owns transfers and checkpoints.
    """
    if not video_key or not output_key:
        raise ValueError("video_key and output_key must be nonempty")
    folder = _local_path(DataFolder.resolve(output_folder).abs_path())
    settings = config or NVENCConfig()

    @describe_builtin(
        "video.transcode_videos",
        video_key=video_key,
        output_key=output_key,
        output_folder=str(folder),
        config=asdict(settings),
    )
    async def _transcode(row: Row) -> Row:
        result = await transcode_video(
            row[video_key],
            folder / f"{uuid.uuid4().hex}.mp4",
            config=settings,
        )
        return row.update({output_key: result.uri})

    return _transcode


def encode_image_sequences(
    *,
    images_key: str,
    output_folder: DataFolderLike,
    fps: float,
    output_key: str = "encoded_video",
    config: NVENCConfig | None = None,
) -> AsyncMapFn:
    """Build an NVENC ``map_async`` block for ordered JPEG/PNG file sequences.

    ``row[images_key]`` is a nonempty list of uniformly sized images, all JPEG or
    all PNG. Its order is the output frame order; no glob or lexical sort is used.
    """
    if not images_key or not output_key:
        raise ValueError("images_key and output_key must be nonempty")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and > 0")
    folder = _local_path(DataFolder.resolve(output_folder).abs_path())
    settings = config or NVENCConfig()

    @describe_builtin(
        "video.encode_image_sequences",
        images_key=images_key,
        output_key=output_key,
        output_folder=str(folder),
        fps=fps,
        config=asdict(settings),
    )
    async def _encode(row: Row) -> Row:
        result = await encode_image_sequence(
            row[images_key],
            folder / f"{uuid.uuid4().hex}.mp4",
            fps=fps,
            config=settings,
        )
        return row.update({output_key: result.uri})

    return _encode
