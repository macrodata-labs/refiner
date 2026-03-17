from __future__ import annotations

from refiner.media import Video


def video_from_timestamp_s(video: Video) -> float:
    return float(video.from_timestamp_s or 0.0)


def video_to_timestamp_s(video: Video) -> float | None:
    return video.to_timestamp_s


def video_uri(video: Video) -> str:
    return str(video.uri)


__all__ = [
    "video_from_timestamp_s",
    "video_to_timestamp_s",
    "video_uri",
]
