from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow

if TYPE_CHECKING:
    from egovision import HandTrackingConfig


def track_hands(
    *,
    video_key: str = "video",
    output_key: str = "hand_tracking",
    config: "HandTrackingConfig | None" = None,
) -> BatchFn:
    """Return a ``batch_map`` function for ego-vision hand tracking.

    Refiner owns row access and passes lazy decoded ``av.VideoFrame`` iterators
    to ``egovision.HandTrackingPipeline``.
    """

    if not video_key:
        raise ValueError("video_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")

    pipeline = None
    episode_input = None

    @describe_builtin(
        "robotics.egocentric:track_hands",
        video_key=video_key,
        output_key=output_key,
    )
    def _track(rows: list[Row]) -> Iterable[Row]:
        nonlocal episode_input, pipeline
        if pipeline is None:
            pipeline, episode_input = _load_egovision(config)

        episodes = []
        for row in rows:
            if not isinstance(row, RoboticsRow) or video_key not in row.videos:
                raise TypeError(
                    "track_hands requires RoboticsRow inputs with video sources"
                )
            episodes.append(
                episode_input(frames=_iter_video_frames(row.videos[video_key]))
            )
        results = pipeline.predict_episodes(episodes)
        if len(results) != len(rows):
            raise ValueError(
                "ego-vision hand tracking returned "
                f"{len(results)} results for {len(rows)} input rows"
            )
        for row, result in zip(rows, results, strict=True):
            yield row.update({output_key: result.to_dict()})

    return _track


def _load_egovision(config: "HandTrackingConfig | None") -> tuple[Any, Any]:
    try:
        from egovision import (
            EpisodeInput,
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
    return HandTrackingPipeline(config), EpisodeInput


def _iter_video_frames(video: Any) -> Iterable[Any]:
    async_frames = video.iter_frames()
    while True:
        try:
            decoded = asyncio.run(anext(async_frames))
        except StopAsyncIteration:
            return
        yield decoded.frame


__all__ = ["track_hands"]
