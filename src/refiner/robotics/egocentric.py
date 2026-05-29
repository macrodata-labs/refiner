from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any

from refiner.execution.asyncio.runtime import submit
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow


def track_hands(
    *,
    video_key: str = "video",
    output_key: str = "hand_tracking",
    config: Any | None = None,
    config_factory: Callable[[Any], Any] | None = None,
) -> BatchFn:
    """Return a ``batch_map`` function for ego-vision hand tracking.

    Refiner owns row access and passes lazy decoded ``av.VideoFrame`` iterators
    to ``egovision.HandTrackingPipeline``.
    """

    if not video_key:
        raise ValueError("video_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")
    if config is not None and config_factory is not None:
        raise ValueError("config and config_factory are mutually exclusive")

    pipeline = None
    episode_input = None

    @describe_builtin(
        "robotics.egocentric:track_hands",
        video_key=video_key,
        output_key=output_key,
        has_config_factory=config_factory is not None,
    )
    def _track(rows: list[Row]) -> Iterable[Row]:
        nonlocal episode_input, pipeline
        if pipeline is None:
            pipeline, episode_input = _load_egovision(config, config_factory)

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


def _load_egovision(
    config: Any | None,
    config_factory: Callable[[Any], Any] | None,
) -> tuple[Any, Any]:
    try:
        egovision = importlib.import_module("egovision")
    except ImportError as exc:
        raise ImportError(
            "track_hands requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        ) from exc

    if config_factory is not None:
        config = config_factory(egovision)
    elif config is None:
        config = egovision.HandTrackingConfig(
            hand_reconstruction=egovision.HaworReconstructionConfig(),
        )
    return egovision.HandTrackingPipeline(config), egovision.EpisodeInput


def _iter_video_frames(video: Any) -> Iterable[Any]:
    async_frames = video.iter_frames()
    while True:
        try:
            decoded = submit(anext(async_frames)).result()
        except StopAsyncIteration:
            return
        yield decoded.frame


__all__ = ["track_hands"]
