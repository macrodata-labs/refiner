from __future__ import annotations

import importlib
import os
import resource
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from refiner.execution.asyncio.runtime import submit
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import BatchFn
from refiner.robotics.row import RoboticsRow
from refiner.worker.context import logger
from refiner.worker.metrics.api import log_gauge


class _HandTrackingPipeline(Protocol):
    def predict_episodes(
        self,
        episodes: list[Any],
        *,
        on_hawor_batch_processed: Callable[[int], None],
    ) -> list[Any]: ...


class _EpisodeInput(Protocol):
    def __call__(self, *, frames: Iterable[Any]) -> Any: ...


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
        Logs ``frames_processed`` as HaWoR model batches finish.
    """

    if not video_key:
        raise ValueError("video_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")

    pipeline: _HandTrackingPipeline | None = None
    episode_input: _EpisodeInput | None = None

    @describe_builtin(
        "robotics.hand_tracking:track_hands",
        video_key=video_key,
        output_key=output_key,
    )
    def _track(rows: list[Row]) -> Iterable[Row]:
        nonlocal episode_input, pipeline
        current_pipeline = pipeline
        current_episode_input = episode_input
        if current_pipeline is None or current_episode_input is None:
            init_start = time.perf_counter()
            logger.info("Initializing ego-vision hand tracking models")
            _log_memory("before_model_init")
            current_pipeline, current_episode_input = _load_egovision(config)
            pipeline = current_pipeline
            episode_input = current_episode_input
            _log_memory("after_model_init")
            logger.info(
                "Initialized ego-vision hand tracking models in "
                f"{time.perf_counter() - init_start:.2f}s"
            )

        episodes = []
        for row in rows:
            if not isinstance(row, RoboticsRow) or video_key not in row.videos:
                raise TypeError(
                    "track_hands requires RoboticsRow inputs with video sources"
                )
            episodes.append(
                current_episode_input(frames=_iter_video_frames(row.videos[video_key]))
            )
        _log_memory("after_episode_inputs")

        def _log_hawor_batch(frame_count: int) -> None:
            if rows:
                rows[0].log_throughput("frames_processed", frame_count, unit="frames")
            _log_memory("after_hawor_batch")

        _log_memory("before_predict_episodes")
        results = current_pipeline.predict_episodes(
            episodes,
            on_hawor_batch_processed=_log_hawor_batch,
        )
        _log_memory("after_predict_episodes")
        if len(results) != len(rows):
            raise ValueError(
                "ego-vision hand tracking returned "
                f"{len(results)} results for {len(rows)} input rows"
            )
        for row, result in zip(rows, results, strict=True):
            hand_tracking = result.to_dict()
            yield row.update({output_key: hand_tracking})
        _log_memory("after_emit_rows")

    return _track


def _load_egovision(config: Any | None) -> tuple[Any, Any]:
    try:
        egovision: Any = importlib.import_module("egovision")
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


def _log_memory(label: str) -> None:
    rss_mb = _rss_mb()
    peak_mb = _peak_rss_mb()
    if rss_mb is not None:
        logger.info(
            f"hand_tracking_memory label={label} rss_mb={rss_mb:.1f} "
            f"peak_rss_mb={peak_mb:.1f}"
        )
        log_gauge("hand_tracking_rss_mb", rss_mb, kind=label, unit="MB")
    if peak_mb is not None:
        log_gauge("hand_tracking_peak_rss_mb", peak_mb, kind=label, unit="MB")


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return None


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


__all__ = ["track_hands"]
