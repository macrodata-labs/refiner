from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Coroutine, Literal

from .types import MediaFile
from .video import DecodedVideo, Video
from .video.utils import decode_video_segment_frames

if TYPE_CHECKING:
    from refiner.pipeline.data.row import Row
else:
    Row = Any


def _coerce_media_value(value: Any) -> Any:
    return MediaFile(value) if isinstance(value, str) else value


def _extract_media(value: Any) -> MediaFile:
    if isinstance(value, MediaFile):
        return value
    media = getattr(value, "media", None)
    if isinstance(media, MediaFile):
        return media
    raise TypeError(
        "hydrate_media expects a string, MediaFile, or object with a `media: MediaFile` attribute"
    )


async def _cache_media(
    media: MediaFile,
    *,
    mode: Literal["file", "bytes"],
    cache_name: str,
    suffix: str | None,
) -> None:
    if mode == "file":
        await media.cache_file(cache_name=cache_name)
    elif media.bytes_cache is None:
        await media.cache_bytes(suffix=suffix, cache_name=cache_name)


async def _decode_video(
    video: Video,
    *,
    cache_name: str,
    decode_backend: Literal["pyav", "ffmpeg"],
) -> Video:
    if isinstance(video.media, DecodedVideo):
        return video

    local_path = await video.media.cache_file(cache_name=cache_name)
    try:
        frames, width, height, pix_fmt = decode_video_segment_frames(
            local_path=local_path,
            from_timestamp_s=float(video.from_timestamp_s or 0.0),
            to_timestamp_s=(
                float(video.to_timestamp_s)
                if video.to_timestamp_s is not None
                else None
            ),
            decoder_cache_name=cache_name,
            decode_backend=decode_backend,
        )
    finally:
        video.media.cleanup()

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
    mode: Literal["file", "bytes"] = "bytes",
    decode: bool = False,
    decode_backend: Literal["pyav", "ffmpeg"] = "pyav",
    on_error: Literal["raise", "null"] = "raise",
    suffix: str | None = None,
    cache_name: str | None = None,
    decoder_cache_name: str | None = None,
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not column:
        raise ValueError("column cannot be empty")
    if mode not in {"file", "bytes"}:
        raise ValueError("mode must be 'file' or 'bytes'")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    target_column = dst_column or column
    resolved_cache_name = cache_name or f"hydrate_media:{column}"
    resolved_decoder_cache_name = decoder_cache_name or resolved_cache_name

    async def _map(row: Row) -> Row:
        try:
            hydrated_value = _coerce_media_value(row[column])

            if isinstance(hydrated_value, Video):
                if decode:
                    hydrated_value = await _decode_video(
                        hydrated_value,
                        cache_name=resolved_decoder_cache_name,
                        decode_backend=decode_backend,
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
                    if isinstance(media, MediaFile):
                        await _cache_media(
                            media,
                            mode=mode,
                            cache_name=resolved_cache_name,
                            suffix=suffix,
                        )
            else:
                await _cache_media(
                    _extract_media(hydrated_value),
                    mode=mode,
                    cache_name=resolved_cache_name,
                    suffix=suffix,
                )
            return row.update({target_column: hydrated_value})
        except Exception:
            if on_error == "raise":
                raise
            return row.update({target_column: None})

    return _map


__all__ = ["hydrate_media"]
