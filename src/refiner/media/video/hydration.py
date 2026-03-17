from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Coroutine, Literal

from refiner.io import DataFile
from refiner.media.video.types import DecodedVideo, Video, VideoFile
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.utils.cache.decoder_cache import get_video_decoder_cache
from refiner.pipeline.utils.cache.file_cache import get_media_cache


async def _decode_video(
    video: Video,
    *,
    cache_name: str,
) -> Video:
    if isinstance(video, DecodedVideo):
        return video

    decoder_cache = get_video_decoder_cache(
        name=f"decode_video:{cache_name}",
        media_cache=get_media_cache(name=f"decode_video:{cache_name}"),
    )
    frames, fps, width, height, pix_fmt = await decoder_cache.decode_segment(
        data_file=DataFile.resolve(video.uri),
        from_timestamp_s=video.from_timestamp_s,
        to_timestamp_s=video.to_timestamp_s,
    )
    if fps is None:
        raise ValueError(f"Failed to resolve FPS for video {video.uri!r}")

    return DecodedVideo(
        frames=frames,
        fps=fps,
        original_file=video,
        width=width,
        height=height,
        pix_fmt=pix_fmt,
    )


def hydrate_video(
    *columns: str,
    on_error: Literal["raise", "null"] = "raise",
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not columns:
        raise ValueError("at least one column is required")
    if any(not column for column in columns):
        raise ValueError("column cannot be empty")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")

    @describe_builtin("video:hydrate_video", columns=list(columns), on_error=on_error)
    async def _map(row: Row) -> Row:
        async def _decode_column(column: str) -> tuple[str, Video | None]:
            try:
                value = row[column]
                if not isinstance(value, (VideoFile, DecodedVideo)):
                    raise ValueError(
                        f"hydrate_video expects a Video for column='{column}'"
                    )
                return (
                    column,
                    await _decode_video(
                        value,
                        cache_name=f"hydrate_video:{column}",
                    ),
                )
            except Exception:
                if on_error == "raise":
                    raise
                return column, None

        updates = dict(
            await asyncio.gather(*(_decode_column(column) for column in columns))
        )
        return row.update(updates)

    return _map


__all__ = ["hydrate_video"]
