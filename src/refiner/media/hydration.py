from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Coroutine, Literal

from refiner.media.video.types import DecodedVideo, Video
from refiner.pipeline.utils.cache.decoder_cache import get_video_decoder_cache
from refiner.pipeline.utils.cache.file_cache import get_media_cache

if TYPE_CHECKING:
    from refiner.pipeline.data.row import Row
else:
    Row = Any


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
        data_file=video.media._data_file,
        from_timestamp_s=float(video.from_timestamp_s or 0.0),
        to_timestamp_s=(
            float(video.to_timestamp_s) if video.to_timestamp_s is not None else None
        ),
    )

    return dataclasses.replace(
        video,
        media=DecodedVideo(
            video_key=video.video_key,
            frames=frames,
            uri=video.uri,
            relative_path=video.relative_path,
            episode_index=video.episode_index,
            frame_index=video.frame_index,
            timestamp_s=video.timestamp_s,
            from_timestamp_s=video.from_timestamp_s,
            to_timestamp_s=video.to_timestamp_s,
            chunk_index=video.chunk_index,
            file_index=video.file_index,
            fps=video.fps,
            width=width,
            height=height,
            pix_fmt=pix_fmt,
        ),
    )


def hydrate_media(
    column: str,
    dst_column: str | None = None,
    *,
    decode: bool = False,
    on_error: Literal["raise", "null"] = "raise",
    cache_name: str | None = None,
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not column:
        raise ValueError("column cannot be empty")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    target_column = dst_column or column
    resolved_cache_name = cache_name or f"hydrate_media:{column}"

    async def _map(row: Row) -> Row:
        try:
            if isinstance(row[column], Video):
                if decode:
                    hydrated_value = await _decode_video(
                        row[column],
                        cache_name=resolved_cache_name,
                    )
                else:
                    raise ValueError(
                        "hydrate_media only supports decoded Video hydration. "
                        "Pass decode=True."
                    )
            else:
                raise ValueError("hydrate_media expects a Video")
            return row.update({target_column: hydrated_value})
        except Exception:
            if on_error == "raise":
                raise
            return row.update({target_column: None})

    return _map


__all__ = ["hydrate_media"]
