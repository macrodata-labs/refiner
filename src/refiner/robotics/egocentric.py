from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow
from refiner.video import VideoSource

if TYPE_CHECKING:
    from egovision import HandTrackingConfig


def track_hands(
    *,
    video_key: str = "video",
    output_key: str = "hand_tracking",
    config: "HandTrackingConfig | None" = None,
) -> BatchFn:
    """Return a ``batch_map`` function for ego-vision hand tracking.

    Refiner owns row access and video decoding. The returned batch function
    materializes each episode as decoded ``av.VideoFrame`` objects, then delegates
    model execution to ``egovision.HandTrackingPipeline``.
    """

    if not video_key:
        raise ValueError("video_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")

    pipeline = None

    @describe_builtin(
        "robotics.egocentric:track_hands",
        video_key=video_key,
        output_key=output_key,
    )
    def _track(rows: list[Row]) -> Iterable[Row]:
        nonlocal pipeline
        if pipeline is None:
            pipeline = _load_pipeline(config)

        episode_input = _load_episode_input()
        episodes = [
            episode_input(
                frames=_decode_video_frames(
                    _row_video_source(
                        row,
                        video_key=video_key,
                    )
                ),
                metadata={},
            )
            for row in rows
        ]
        results = pipeline.predict_episodes(episodes)
        if len(results) != len(rows):
            raise ValueError(
                "ego-vision hand tracking returned "
                f"{len(results)} results for {len(rows)} input rows"
            )
        for row, result in zip(rows, results, strict=True):
            yield row.update({output_key: _plain_result(result)})

    return _track


def _load_pipeline(config: "HandTrackingConfig | None") -> Any:
    try:
        from egovision import (
            HandTrackingConfig,
            HandTrackingPipeline,
            HaworReconstructionConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "track_hands requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        ) from exc

    if config is None:
        config = HandTrackingConfig(
            hand_reconstruction=HaworReconstructionConfig(),
        )
    return HandTrackingPipeline(config)


def _load_episode_input() -> Any:
    try:
        from egovision import EpisodeInput
    except ImportError as exc:
        raise ImportError(
            "track_hands requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        ) from exc

    return EpisodeInput


def _row_video_source(
    row: Row,
    *,
    video_key: str,
) -> VideoSource:
    if isinstance(row, RoboticsRow) and video_key in row.videos:
        return row.videos[video_key]
    raise TypeError("track_hands requires RoboticsRow inputs with video sources")


def _decode_video_frames(video: VideoSource) -> list[Any]:
    async def _collect() -> list[Any]:
        return [decoded.frame async for decoded in video.iter_frames()]

    frames = asyncio.run(_collect())
    if not frames:
        raise ValueError("video source did not produce any frames")
    return frames


def _plain_result(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return _plain(value)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = ["track_hands"]
