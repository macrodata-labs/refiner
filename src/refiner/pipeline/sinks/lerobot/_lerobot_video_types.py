from __future__ import annotations

from refiner.media import VideoFile


def video_from_timestamp_s(video: VideoFile) -> float:
    return float(video.from_timestamp_s or 0.0)


def video_to_timestamp_s(video: VideoFile) -> float | None:
    return video.to_timestamp_s


def video_uri(video: VideoFile) -> str:
    return str(video.uri)


__all__ = [
    "video_from_timestamp_s",
    "video_to_timestamp_s",
    "video_uri",
]
