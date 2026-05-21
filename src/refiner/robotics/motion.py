from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.row import RoboticsRow


def _motion_energy(values: object) -> np.ndarray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = values.to_pylist()
    values = np.asarray(values, dtype=np.float32)
    values = values.reshape(-1, 1) if values.ndim == 1 else values
    energy = np.zeros(len(values), dtype=np.float32)
    if len(values) <= 1:
        return energy

    # energy[t] = mean(abs(values[t] - values[t-1]))
    deltas = np.abs(np.diff(values, axis=0))
    energy[1:] = deltas.mean(axis=1)
    return energy


def motion_trim(
    *,
    threshold: float = 0.001,
    pad_frames: int = 0,
    timestamp_key: str = "timestamp",
    action_key: str = "action",
    state_key: str | Sequence[str] = "observation.state",
) -> Callable[[Row], Row]:
    """Return a row mapper that trims robotics episodes to the active motion window.

    The trim span is inferred from the earliest action/state activity above the
    threshold and then applied to the episode frame data. Video timestamps are
    clipped to the same span, and stale video stats are dropped so writers
    recompute them.
    """

    if not action_key:
        raise ValueError("action_key cannot be empty")
    if not state_key:
        raise ValueError("state_key cannot be empty")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if pad_frames < 0:
        raise ValueError("pad_frames must be >= 0")
    if not timestamp_key:
        raise ValueError("timestamp_key cannot be empty")

    @describe_builtin(
        "robotics:motion_trim",
        threshold=threshold,
        pad_frames=pad_frames,
        timestamp_key=timestamp_key,
        action_key=action_key,
        state_key=state_key,
    )
    def _trim(row: Row) -> Row:
        if not isinstance(row, RoboticsRow):
            raise ValueError("motion_trim requires RoboticsRow inputs")

        if row.num_frames <= 0:
            raise ValueError(
                "motion_trim requires robotics rows with non-empty frames "
                f"containing '{timestamp_key}', '{action_key}', and '{state_key}'"
            )
        if row.shard_id is not None:
            row.log_throughput("frames_in", row.num_frames, unit="frames")

        try:
            frame_table = row.to_frame_table()
            timestamp_values = row.timestamps
            if timestamp_values is None:
                timestamp_values = frame_table.column(timestamp_key)
            action_values = row.actions
            if action_values is None:
                action_values = frame_table.column(action_key)
            state_values = row.states
            if state_values is None:
                if not isinstance(state_key, str):
                    raise KeyError(state_key)
                state_values = frame_table.column(state_key)
        except KeyError as exc:
            raise ValueError(
                "motion_trim requires robotics rows with non-empty frames "
                f"containing '{timestamp_key}', '{action_key}', and '{state_key}'"
            ) from exc

        # Normalize timestamps to a dense float array for window selection.
        timestamps = (
            np.asarray(
                timestamp_values.to_numpy(zero_copy_only=False), dtype=np.float32
            )
            if isinstance(timestamp_values, (pa.Array, pa.ChunkedArray))
            else np.asarray(timestamp_values, dtype=np.float32)
        ).reshape(-1)

        # compute motion energy for actions and state
        action_active = np.flatnonzero(_motion_energy(action_values) > threshold)
        state_active = np.flatnonzero(_motion_energy(state_values) > threshold)
        if action_active.size == 0 and state_active.size == 0:
            if row.shard_id is not None:
                row.log_throughput("episodes_fully_trimmed", 1, unit="episodes")
                row.log_throughput("frames_removed", row.num_frames, unit="frames")
                row.log_histogram("trim_fraction", 1.0, unit="ratio")
            return cast(Row, row.select_frames([]))

        start_candidates = [
            int(active[0])
            for active in (action_active, state_active)
            if active.size > 0
        ]
        end_candidates = [
            int(active[-1])
            for active in (action_active, state_active)
            if active.size > 0
        ]
        start_idx = max(0, min(start_candidates) - 1 - pad_frames)
        end_idx = min(row.num_frames - 1, max(end_candidates) + pad_frames)
        keep_indices = list(range(start_idx, end_idx + 1))
        kept_start_ts = float(timestamps[start_idx])
        removed_frames = row.num_frames - len(keep_indices)
        if row.shard_id is not None:
            if removed_frames > 0:
                row.log_throughput("episodes_trimmed", 1, unit="episodes")
                row.log_throughput("frames_removed", removed_frames, unit="frames")
            row.log_throughput("frames_out", len(keep_indices), unit="frames")
            row.log_histogram(
                "trim_fraction",
                removed_frames / max(1, row.num_frames),
                unit="ratio",
            )
        kept_duration_s = (
            float(timestamps[end_idx + 1]) - kept_start_ts
            if end_idx + 1 < row.num_frames
            else None
        )
        trimmed = row.select_frames(keep_indices)
        shifted_timestamps = (timestamps[keep_indices] - kept_start_ts).tolist()
        if trimmed.timestamps is not None:
            trimmed = trimmed.with_timestamps(shifted_timestamps)
        else:
            frame_table = trimmed.to_frame_table()
            value_type = (
                frame_table.table.schema.field(timestamp_key).type
                if timestamp_key in frame_table.names
                else None
            )
            trimmed = trimmed.update(
                frames=Tabular(
                    set_or_append_column(
                        frame_table.table,
                        timestamp_key,
                        pa.array(shifted_timestamps, type=value_type),
                    )
                )
            )
        for video_key, video in row.videos.items():
            trimmed_to_ts = kept_duration_s
            trimmed = trimmed.with_video(
                video_key,
                video.clipped(
                    from_timestamp_s=kept_start_ts,
                    to_timestamp_s=trimmed_to_ts,
                ),
            )
            trimmed = trimmed.drop_stats(video_key)

        return cast(Row, trimmed)

    return _trim


__all__ = ["motion_trim"]
