from __future__ import annotations

import importlib
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from refiner.execution.asyncio.runtime import submit
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow
from refiner.worker.context import logger

HAND_SIDES: tuple[str, ...] = ("left", "right")
CAMERA_HAND_FIELDS: dict[str, tuple[int, ...]] = {
    "joints_camera": (21, 3),
    "T_camera_wrist": (4, 4),
    "mano_pose": (96,),
    "mano_shape": (10,),
    "mano_translation": (3,),
    "confidence": (),
    "infilled": (),
}
WORLD_HAND_FIELDS: dict[str, tuple[int, ...]] = {
    "joints_world": (21, 3),
    "T_world_wrist": (4, 4),
    "mano_pose": (96,),
    "mano_shape": (10,),
    "confidence": (),
    "infilled": (),
}


def track_hands(
    *,
    video_key: str,
    output_key: str = "hand_tracking",
    config: Any | None = None,
) -> BatchFn:
    """Return a ``batch_map`` function that annotates videos with hand tracks.

    Input rows must be ``RoboticsRow`` values with a video source available at
    ``row.videos[video_key]``. Use ``Pipeline.to_robot_rows(video_keys=...)`` to
    convert a URL, local path, or video asset column into that robotics view. The
    video source must expose ``iter_frames()``, yielding decoded frames that can
    be converted to RGB by ego-vision.

    Args:
        video_key: Required key in ``row.videos`` to process.
        output_key: Row column that will receive the hand-tracking payload.
            Defaults to ``"hand_tracking"``.
        config: Optional ``egovision.HandTrackingConfig``. If omitted, Refiner
            constructs the default ego-vision hand-tracking pipeline with HaWoR
            reconstruction.

    Returns:
        A ``BatchFn`` for ``Pipeline.batch_map``. Each output row is the input
        row plus ``row[output_key]``, a dictionary with:

        ``episode_id``:
            Ego-vision episode identifier.
        ``camera_trajectory``:
            ``numpy.ndarray`` of per-frame world-from-camera transforms,
            shaped ``[T, 4, 4]``.
        ``intrinsics``:
            ``numpy.ndarray`` of per-frame camera intrinsics, usually shaped
            ``[T, 3, 3]``.
        ``hands_camera``:
            Dictionary keyed by ``"left"`` and ``"right"``. Each hand entry has
            ``joints_camera`` ``[T, 21, 3]``, ``T_camera_wrist`` ``[T, 4, 4]``,
            ``mano_pose`` ``[T, 96]``, ``mano_shape`` ``[T, 10]``,
            ``mano_translation`` ``[T, 3]``, ``confidence`` ``[T]``, and
            ``infilled`` ``[T]``.
        ``hands_world``:
            Dictionary keyed by ``"left"`` and ``"right"``. Each hand entry has
            ``joints_world`` ``[T, 21, 3]``, ``T_world_wrist`` ``[T, 4, 4]``,
            ``mano_pose`` ``[T, 96]``, ``mano_shape`` ``[T, 10]``, ``confidence``
            ``[T]``, and ``infilled`` ``[T]``.
        ``metadata``:
            Optional dictionary with pipeline settings such as
            ``vggt_seq_length`` and ``hawor_seq_length``.

    Metrics:
        Logs ``egovision_frames_decoded`` while frames are read,
        ``frames_processed`` from the resulting hand-track frame count, and
        ``egovision_episodes_processed`` once an episode has been annotated.
    """

    if not video_key:
        raise ValueError("video_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")

    pipeline = None
    episode_input = None

    @describe_builtin(
        "robotics.hand_tracking:track_hands",
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
            raw_hand_tracking = result.to_dict()
            hawor_frame_count = _hand_tracking_frame_count(raw_hand_tracking)
            if hawor_frame_count <= 0:
                hawor_frame_count = decoded_frame_count
            hand_tracking = _normalize_hand_tracking_payload(raw_hand_tracking)
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
            "`pip install macrodata-refiner[hand_tracking]`."
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


def _normalize_hand_tracking_payload(hand_tracking: dict[str, Any]) -> dict[str, Any]:
    frame_count = _tracking_frame_count(hand_tracking)
    payload = dict(hand_tracking)
    payload["camera_trajectory"] = _array_or_empty(
        payload.get("camera_trajectory"), frame_count, (4, 4)
    )
    payload["intrinsics"] = _array_or_empty(
        payload.get("intrinsics"), frame_count, (3, 3)
    )
    payload["hands_camera"] = _normalize_hands(
        payload.get("hands_camera"),
        frame_count,
        CAMERA_HAND_FIELDS,
    )
    payload["hands_world"] = _normalize_hands(
        payload.get("hands_world"),
        frame_count,
        WORLD_HAND_FIELDS,
    )
    return payload


def _tracking_frame_count(hand_tracking: dict[str, Any]) -> int:
    for key in ("camera_trajectory", "intrinsics"):
        value = hand_tracking.get(key)
        if value is not None:
            return len(value)
    return max(
        _hand_tracking_frame_count(hand_tracking),
        _hand_group_frame_count(hand_tracking.get("hands_camera")),
    )


def _hand_group_frame_count(hands: Any) -> int:
    if isinstance(hands, dict):
        return max((_hand_frame_count(hand) for hand in hands.values()), default=0)
    if isinstance(hands, list):
        return max((_hand_frame_count(hand) for hand in hands), default=0)
    return 0


def _normalize_hands(
    hands: Any,
    frame_count: int,
    field_shapes: dict[str, tuple[int, ...]],
) -> dict[str, dict[str, np.ndarray]]:
    hands_by_side = _hands_by_side(hands)
    return {
        side: {
            field: _hand_field_array(
                hands_by_side.get(side, {}).get(field),
                frame_count,
                shape,
                field=field,
            )
            for field, shape in field_shapes.items()
        }
        for side in HAND_SIDES
    }


def _hands_by_side(hands: Any) -> dict[str, dict[str, Any]]:
    if isinstance(hands, dict):
        return {side: hand for side, hand in hands.items() if isinstance(hand, dict)}
    if isinstance(hands, list):
        return {
            hand["handedness"]: hand
            for hand in hands
            if isinstance(hand, dict) and hand.get("handedness") in HAND_SIDES
        }
    return {}


def _hand_field_array(
    value: Any,
    frame_count: int,
    frame_shape: tuple[int, ...],
    *,
    field: str,
) -> np.ndarray:
    if value is None:
        return _empty_hand_field(frame_count, frame_shape, field=field)
    array = np.asarray(value)
    if len(array) == frame_count:
        return array
    filled = _empty_hand_field(frame_count, frame_shape, field=field)
    count = min(frame_count, len(array))
    if count:
        filled[:count] = array[:count]
    return filled


def _empty_hand_field(
    frame_count: int,
    frame_shape: tuple[int, ...],
    *,
    field: str,
) -> np.ndarray:
    if field == "infilled":
        return np.zeros((frame_count, *frame_shape), dtype=bool)
    if field == "confidence":
        return np.zeros((frame_count, *frame_shape), dtype=np.float64)
    return np.full((frame_count, *frame_shape), np.nan, dtype=np.float64)


def _array_or_empty(
    value: Any,
    frame_count: int,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    if value is None:
        return np.full((frame_count, *frame_shape), np.nan, dtype=np.float64)
    array = np.asarray(value)
    if len(array) == frame_count:
        return array
    filled = np.full((frame_count, *frame_shape), np.nan, dtype=np.float64)
    count = min(frame_count, len(array))
    if count:
        filled[:count] = array[:count]
    return filled


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
