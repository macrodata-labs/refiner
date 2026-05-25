from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.pipeline.steps import FlatMapFn, MapResult


HAND_TRACKING_FLUSH_COLUMN = "__refiner_hand_tracking_flush__"


HandTrackingBatchFn = Callable[[list[Row]], Iterable[MapResult]]
HandTrackingFlushPredicate = Callable[[Row], bool]


def run_hand_tracking(
    fn: HandTrackingBatchFn,
    *,
    batch_size: int,
    flush_when: HandTrackingFlushPredicate | None = None,
    include_flush_row: bool = False,
) -> FlatMapFn:
    """Build a buffered ``flat_map`` function for episode-level hand tracking.

    The returned function buffers episode rows and calls ``fn(batch)`` when the
    episode batch is full. It is meant for pipelines that keep one row per
    episode/video but still want to run fixed-size model batches inside the
    transform.

    Put model/runtime state in the closure captured by ``fn``. For example,
    initialize models lazily inside ``fn`` and reuse them across later calls.

    A flush sentinel can be handled by passing
    ``flush_when=is_hand_tracking_flush_row``. By default, that sentinel is not
    included in the model batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    buffer: list[Row] = []

    def apply(row: Row) -> Iterable[MapResult]:
        should_flush = bool(flush_when(row)) if flush_when is not None else False
        if not should_flush or include_flush_row:
            buffer.append(row)

        if not should_flush and len(buffer) < batch_size:
            return []

        return _run_buffered_batch(fn, buffer)

    return apply


def track_hands_egovision(
    *,
    config: Any | None = None,
    frames_key: str = "frames",
    output_key: str = "hand_tracking",
    metadata_keys: Iterable[str] = (),
) -> Callable[[list[Row]], Iterable[Row]]:
    """Return a ``batch_map`` function backed by ``egovision`` hand tracking.

    Rows are expected to be episode/video rows. ``frames_key`` must contain an
    iterable of decoded ``av.VideoFrame``-compatible objects. Refiner owns video
    decoding; ego-vision owns model scheduling and prediction.
    """

    if not frames_key:
        raise ValueError("frames_key cannot be empty")
    if not output_key:
        raise ValueError("output_key cannot be empty")

    selected_metadata_keys = tuple(metadata_keys)
    pipeline: Any | None = None

    @describe_builtin(
        "robotics.egocentric:track_hands_egovision",
        frames_key=frames_key,
        output_key=output_key,
    )
    def _track(rows: list[Row]) -> Iterable[Row]:
        nonlocal pipeline
        if pipeline is None:
            pipeline = _load_egovision_pipeline(config)

        episode_input_type = _load_egovision_episode_input()
        episodes = [
            episode_input_type(
                frames=row[frames_key],
                metadata=_row_metadata(row, selected_metadata_keys),
            )
            for row in rows
        ]
        results = pipeline.predict_episodes(episodes)
        if len(results) != len(rows):
            msg = (
                "ego-vision hand tracking returned "
                f"{len(results)} results for {len(rows)} rows"
            )
            raise ValueError(msg)
        for row, result in zip(rows, results, strict=True):
            yield row.update({output_key: _egovision_result_to_dict(result)})

    return _track


def hand_tracking_flush_row(**values: Any) -> dict[str, Any]:
    """Create a sentinel row that flushes ``run_hand_tracking`` buffers."""

    return {HAND_TRACKING_FLUSH_COLUMN: True, **values}


def is_hand_tracking_flush_row(row: Row) -> bool:
    """Return whether a row is the hand-tracking flush sentinel."""

    return bool(row.get(HAND_TRACKING_FLUSH_COLUMN, False))


def _run_buffered_batch(
    fn: HandTrackingBatchFn,
    buffer: list[Row],
) -> list[MapResult]:
    if not buffer:
        return []
    batch = list(buffer)
    buffer.clear()
    return list(fn(batch))


def _load_egovision_pipeline(config: Any | None) -> Any:
    try:
        from egovision import (
            HandTrackingConfig,
            HandTrackingPipeline,
            HaworReconstructionConfig,
        )
    except ImportError as exc:
        msg = (
            "track_hands_egovision requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        )
        raise ImportError(msg) from exc

    resolved_config = config
    if resolved_config is None:
        resolved_config = HandTrackingConfig(
            hand_reconstruction=HaworReconstructionConfig(),
        )
    return HandTrackingPipeline(resolved_config)


def _load_egovision_episode_input() -> Any:
    try:
        from egovision import EpisodeInput
    except ImportError as exc:
        msg = (
            "track_hands_egovision requires ego-vision. Install it with "
            "`pip install macrodata-refiner[egocentric]`."
        )
        raise ImportError(msg) from exc
    return EpisodeInput


def _row_metadata(row: Row, metadata_keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key] for key in metadata_keys if key in row}


def _egovision_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if is_dataclass(result) and not isinstance(result, type):
        result = asdict(result)
    if not isinstance(result, dict):
        msg = "ego-vision hand tracking result must be a mapping or expose to_dict()"
        raise TypeError(msg)
    return _plain_python(result)


def _plain_python(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_python(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_python(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = [
    "HAND_TRACKING_FLUSH_COLUMN",
    "HandTrackingBatchFn",
    "HandTrackingFlushPredicate",
    "hand_tracking_flush_row",
    "is_hand_tracking_flush_row",
    "run_hand_tracking",
    "track_hands_egovision",
]
