from __future__ import annotations

from refiner.media import Video
from refiner.media.video.types import DecodedVideo


def video_from_timestamp_s(video: Video) -> float:
    if isinstance(video, DecodedVideo):
        return 0.0
    return float(video.from_timestamp_s or 0.0)


def video_to_timestamp_s(video: Video) -> float | None:
    if isinstance(video, DecodedVideo):
        return float(len(video.frames) * (1.0 / video.fps))
    return video.to_timestamp_s


def video_uri(video: Video) -> str:
    if isinstance(video, DecodedVideo):
        return str(video.original_file.uri)
    return str(video.uri)


__all__ = [
    "video_from_timestamp_s",
    "video_to_timestamp_s",
    "video_uri",
]
