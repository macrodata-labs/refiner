from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Coroutine, Literal

from .types import MediaFile
from .video import DecodedVideo, Video
from .video.utils import decode_video_segment_frames

if TYPE_CHECKING:
    from refiner.sources.row import Row
else:
    Row = Any


def _coerce_media_value(value: Any) -> Any:
    if isinstance(value, str):
        return MediaFile(value)
    return value


def _extract_media(value: Any) -> MediaFile:
    if isinstance(value, MediaFile):
        return value
    media = getattr(value, "media", None)
    if isinstance(media, MediaFile):
        return media
    raise TypeError(
        "hydrate_media expects a string, MediaFile, or object with a `media: MediaFile` attribute"
    )


def hydrate_media(
    column: str,
    dst_column: str | None = None,
    *,
    decode: bool = False,
    decode_backend: Literal["pyav", "ffmpeg"] = "pyav",
    on_error: Literal["raise", "null"] = "raise",
    suffix: str | None = None,
    cache_name: str | None = None,
    decoder_cache_name: str | None = None,
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not column:
        raise ValueError("column cannot be empty")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    target_column = dst_column or column
    resolved_cache_name = cache_name or f"hydrate_media:{column}"
    resolved_decoder_cache_name = decoder_cache_name or resolved_cache_name

    async def _map(row: Row) -> Row:
        try:
            hydrated_value = _coerce_media_value(row[column])
            loop = asyncio.get_running_loop()

            if isinstance(hydrated_value, Video):
                if decode:
                    if isinstance(hydrated_value.media, DecodedVideo):
                        return row.update({target_column: hydrated_value})
                    def _decode_video_in_cache():
                        with hydrated_value.media.cached_path(
                            suffix=".mp4",
                            cache_name=resolved_cache_name,
                        ) as local_path:
                            return decode_video_segment_frames(
                                local_path=local_path,
                                from_timestamp_s=float(
                                    hydrated_value.from_timestamp_s or 0.0
                                ),
                                to_timestamp_s=(
                                    float(hydrated_value.to_timestamp_s)
                                    if hydrated_value.to_timestamp_s is not None
                                    else None
                                ),
                                decoder_cache_name=resolved_decoder_cache_name,
                                decode_backend=decode_backend,
                            )

                    frames, width, height, pix_fmt = await loop.run_in_executor(
                        None, _decode_video_in_cache
                    )
                    hydrated_value = dataclasses.replace(
                        hydrated_value,
                        media=DecodedVideo(
                            video_key=hydrated_value.video_key,
                            frames=frames,
                            uri=hydrated_value.uri,
                            relative_path=hydrated_value.relative_path,
                            episode_index=hydrated_value.episode_index,
                            frame_index=hydrated_value.frame_index,
                            timestamp_s=hydrated_value.timestamp_s,
                            from_timestamp_s=hydrated_value.from_timestamp_s,
                            to_timestamp_s=hydrated_value.to_timestamp_s,
                            chunk_index=hydrated_value.chunk_index,
                            file_index=hydrated_value.file_index,
                            fps=hydrated_value.fps,
                            width=width,
                            height=height,
                            pix_fmt=pix_fmt,
                        ),
                    )
                else:
                    if not isinstance(hydrated_value.media, DecodedVideo) and (
                        hydrated_value.from_timestamp_s is not None
                        or hydrated_value.to_timestamp_s is not None
                    ):
                        raise ValueError(
                            "Cannot hydrate timestamped Video with decode=False. "
                            "Pass decode=True when hydrate_media is called."
                        )
                    media = hydrated_value.media
                    if isinstance(media, MediaFile) and media.bytes_cache is None:
                        await loop.run_in_executor(
                            None,
                            partial(
                                media.cache_bytes,
                                suffix=suffix,
                                cache_name=resolved_cache_name,
                            ),
                        )
            else:
                media = _extract_media(hydrated_value)
                if media.bytes_cache is None:
                    await loop.run_in_executor(
                        None,
                        partial(
                            media.cache_bytes,
                            suffix=suffix,
                            cache_name=resolved_cache_name,
                        ),
                    )
            return row.update({target_column: hydrated_value})
        except Exception:
            if on_error == "raise":
                raise
            return row.update({target_column: None})

    return _map


__all__ = ["hydrate_media"]
