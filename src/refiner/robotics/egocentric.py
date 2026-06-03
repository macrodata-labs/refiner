from __future__ import annotations

import importlib
import time
from collections.abc import Iterable
from typing import Any

from refiner.execution.asyncio.runtime import submit
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow
from refiner.worker.context import logger


def track_hands(
    *,
    video_key: str = "video",
    output_key: str = "hand_tracking",
    config: Any | None = None,
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
            init_start = time.perf_counter()
            logger.info("Initializing ego-vision hand tracking models")
            pipeline, episode_input = _load_egovision(config)
            logger.info(
                "Initialized ego-vision hand tracking models in "
                f"{time.perf_counter() - init_start:.2f}s"
            )

        episodes = []
        frame_counts = [0] * len(rows)
        for row_index, row in enumerate(rows):
            if not isinstance(row, RoboticsRow) or video_key not in row.videos:
                raise TypeError(
                    "track_hands requires RoboticsRow inputs with video sources"
                )
            episodes.append(
                episode_input(
                    frames=_count_frames(
                        _iter_video_frames(row.videos[video_key]),
                        row=row,
                        frame_counts=frame_counts,
                        row_index=row_index,
                    )
                )
            )
        results = pipeline.predict_episodes(episodes)
        if len(results) != len(rows):
            raise ValueError(
                "ego-vision hand tracking returned "
                f"{len(results)} results for {len(rows)} input rows"
            )
        for row, result, decoded_frame_count in zip(
            rows, results, frame_counts, strict=True
        ):
            hand_tracking = result.to_dict()
            hawor_frame_count = _hand_tracking_frame_count(hand_tracking)
            if hawor_frame_count <= 0:
                hawor_frame_count = decoded_frame_count
            row.log_throughput("frames_processed", hawor_frame_count, unit="frames")
            row.log_throughput("egovision_episodes_processed", 1, unit="episodes")
            yield row.update({output_key: hand_tracking})

    return _track


def _load_egovision(config: Any | None) -> tuple[Any, Any]:
    try:
        egovision = importlib.import_module("egovision")
    except ImportError as exc:
        raise ImportError(
            "track_hands requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        ) from exc

    if config is None:
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


def _count_frames(
    frames: Iterable[Any],
    *,
    row: Row,
    frame_counts: list[int],
    row_index: int,
) -> Iterable[Any]:
    for frame in frames:
        frame_counts[row_index] += 1
        row.log_throughput("egovision_frames_decoded", 1, unit="frames")
        yield frame


def _hand_tracking_frame_count(hand_tracking: Any) -> int:
    hands_world = (
        hand_tracking.get("hands_world") if isinstance(hand_tracking, dict) else None
    )
    if isinstance(hands_world, dict):
        return max(
            (_hand_frame_count(hand) for hand in hands_world.values()), default=0
        )
    if isinstance(hands_world, list):
        return max((_hand_frame_count(hand) for hand in hands_world), default=0)
    return 0


def _hand_frame_count(hand: Any) -> int:
    if not isinstance(hand, dict):
        return 0
    for key in ("confidence", "joints_world", "T_world_wrist", "mano_pose"):
        value = hand.get(key)
        if value is not None:
            return len(value)
    return 0


__all__ = ["track_hands"]
