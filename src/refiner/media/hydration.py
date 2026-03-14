from __future__ import annotations

from collections.abc import Callable
from typing import Any, Coroutine, Literal

from refiner.io import DataFile
from refiner.pipeline.data.row import Row
from refiner.media.video.types import DecodedVideo, Video
from refiner.pipeline.utils.cache.decoder_cache import get_video_decoder_cache
from refiner.pipeline.utils.cache.file_cache import get_media_cache


async def _decode_video(
    video: Video,
    *,
    cache_name: str,
) -> Video:
    if isinstance(video.media, DecodedVideo):
        return video

    decoder_cache = get_video_decoder_cache(
        name=f"decode_video:{cache_name}",
        media_cache=get_media_cache(name=f"decode_video:{cache_name}"),
    )
    frames, width, height, pix_fmt = await decoder_cache.decode_segment(
        data_file=DataFile.resolve(video.uri),
        from_timestamp_s=video.from_timestamp_s,
        to_timestamp_s=video.to_timestamp_s,
    )

    return Video(
        media=DecodedVideo(
            frames=frames,
            uri=video.uri,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
        ),
        from_timestamp_s=video.from_timestamp_s,
        to_timestamp_s=video.to_timestamp_s,
    )


def hydrate_media(
    column: str,
    dst_column: str | None = None,
    *,
    on_error: Literal["raise", "null"] = "raise",
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not column:
        raise ValueError("column cannot be empty")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    target_column = dst_column or column
    cache_name = f"hydrate_media:{column}"

    async def _map(row: Row) -> Row:
        try:
            value = row[column]
            if not isinstance(value, Video):
                raise ValueError("hydrate_media expects a Video")
            hydrated_value = await _decode_video(value, cache_name=cache_name)
            return row.update({target_column: hydrated_value})
        except Exception:
            if on_error == "raise":
                raise
            return row.update({target_column: None})

    return _map


__all__ = ["hydrate_media"]
