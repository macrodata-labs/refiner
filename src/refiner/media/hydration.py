from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Coroutine, Literal

from .types import MediaFile

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
    mode: Literal["file", "bytes"] = "file",
    on_error: Literal["raise", "null"] = "raise",
    suffix: str | None = None,
) -> Callable[[Row], Coroutine[Any, Any, Row]]:
    if not column:
        raise ValueError("column cannot be empty")
    if mode not in {"file", "bytes"}:
        raise ValueError("mode must be 'file' or 'bytes'")
    if on_error not in {"raise", "null"}:
        raise ValueError("on_error must be 'raise' or 'null'")
    target_column = dst_column or column

    async def _map(row: Row) -> Row:
        try:
            hydrated_value = _coerce_media_value(row[column])
            media = _extract_media(hydrated_value)
            loop = asyncio.get_running_loop()
            if mode == "file" and media.local_path is None:
                await loop.run_in_executor(
                    None, partial(media.cache_locally, suffix=suffix)
                )
            elif mode == "bytes" and media.bytes_cache is None:
                await loop.run_in_executor(None, media.cache_bytes)
            return row.update({target_column: hydrated_value})
        except Exception:
            if on_error == "raise":
                raise
            return row.update({target_column: None})

    return _map


__all__ = ["hydrate_media"]
