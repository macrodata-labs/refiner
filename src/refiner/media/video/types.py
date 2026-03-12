from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from refiner.media.types import MediaFile


@dataclass(frozen=True, slots=True)
class Video:
    media: "MediaFile | DecodedVideo"
    video_key: str
    relative_path: str | None = None
    episode_index: int | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None
    chunk_index: int | None = None
    file_index: int | None = None
    fps: int | None = None

    @property
    def uri(self) -> str:
        return self.media.uri

    @property
    def bytes_cache(self) -> bytes | None:
        return getattr(self.media, "bytes_cache", None)


@dataclass(frozen=True, slots=True)
class VideoBytes:
    video: Video
    from_timestamp_s: float
    to_timestamp_s: float | None


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    """A hydrated video payload represented as raw decoded frames and metadata."""

    video_key: str
    frames: tuple[np.ndarray, ...]
    uri: str
    relative_path: str | None = None
    episode_index: int | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    from_timestamp_s: float | None = None
    to_timestamp_s: float | None = None
    chunk_index: int | None = None
    file_index: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = "rgb24"

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float | None:
        if self.fps is None or self.fps <= 0:
            return None
        return float(self.frame_count) / float(self.fps)

    @property
    def bytes_cache(self) -> bytes | None:
        return None

    @classmethod
    def from_video(
        cls,
        *,
        video: Video,
        frames: tuple[np.ndarray, ...],
    ) -> "DecodedVideo":
        first_frame = frames[0] if frames else None
        height = None
        width = None
        if isinstance(first_frame, np.ndarray) and first_frame.ndim >= 2:
            height = int(first_frame.shape[0])
            width = int(first_frame.shape[1])

        return cls(
            video_key=video.video_key,
            frames=frames,
            uri=video.uri,
            width=width,
            height=height,
            relative_path=video.relative_path,
            episode_index=video.episode_index,
            frame_index=video.frame_index,
            timestamp_s=video.timestamp_s,
            from_timestamp_s=video.from_timestamp_s,
            to_timestamp_s=video.to_timestamp_s,
            chunk_index=video.chunk_index,
            file_index=video.file_index,
            fps=video.fps,
            pix_fmt="rgb24",
        )


__all__ = ["Video", "DecodedVideo"]
