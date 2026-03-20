from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.lerobot_format import LeRobotRow


def _motion_energy(values: np.ndarray | list[object]) -> np.ndarray:
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
    action_key: str = "action",
    state_key: str = "observation.state",
) -> Callable[[Row], Row]:
    """Return a row mapper that trims LeRobot episodes to the active motion window.

    The trim span is inferred from the earliest action/state activity above the
    threshold and then applied to the episode frame list. LeRobot video timestamp
    fields are updated on the row itself, and stale video stats are dropped so
    downstream sinks recompute them.
    """

    if not action_key:
        raise ValueError("action_key cannot be empty")
    if not state_key:
        raise ValueError("state_key cannot be empty")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if pad_frames < 0:
        raise ValueError("pad_frames must be >= 0")

    @describe_builtin(
        "robotics:motion_trim",
        threshold=threshold,
        pad_frames=pad_frames,
        action_key=action_key,
        state_key=state_key,
    )
    def _trim(row: Row) -> Row:
        if not isinstance(row, LeRobotRow):
            raise ValueError("motion_trim requires LeRobotRow inputs")

        frames = (
            row.frames
            if isinstance(row.frames, Tabular)
            else Tabular.from_rows(row.frames)
        )
        frame_table = frames.table
        if frame_table.num_rows <= 0 or any(
            key not in frame_table.column_names
            for key in ("timestamp", action_key, state_key)
        ):
            raise ValueError(
                "motion_trim requires LeRobot-format rows with non-empty 'frames' "
                f"containing 'timestamp', '{action_key}', and '{state_key}'"
            )
        if row.shard_id is not None:
            row.log_throughput("frames_in", frames.num_rows, unit="frames")

        # timestamps
        timestamp_column = frame_table.column("timestamp")
        timestamps = np.asarray(
            timestamp_column.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(-1)

        # compute motion energy for actions and state
        action_active = np.flatnonzero(
            _motion_energy(frame_table.column(action_key).to_pylist()) > threshold
        )
        state_active = np.flatnonzero(
            _motion_energy(frame_table.column(state_key).to_pylist()) > threshold
        )
        if action_active.size == 0 and state_active.size == 0:
            if row.shard_id is not None:
                row.log_throughput("episodes_fully_trimmed", 1, unit="episodes")
                row.log_throughput("frames_removed", frames.num_rows, unit="frames")
                row.log_histogram("trim_fraction", 1.0, unit="ratio")
            return row.update(frames=[])

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
        end_idx = min(frame_table.num_rows - 1, max(end_candidates) + pad_frames)
        kept_table = frame_table.slice(start_idx, end_idx - start_idx + 1)
        kept_start_ts = float(timestamps[start_idx])
        removed_frames = frame_table.num_rows - kept_table.num_rows
        if row.shard_id is not None:
            if removed_frames > 0:
                row.log_throughput("episodes_trimmed", 1, unit="episodes")
                row.log_throughput("frames_removed", removed_frames, unit="frames")
            row.log_throughput("frames_out", kept_table.num_rows, unit="frames")
            row.log_histogram(
                "trim_fraction",
                removed_frames / max(1, frame_table.num_rows),
                unit="ratio",
            )
        kept_duration_s = (
            float(timestamps[end_idx + 1]) - kept_start_ts
            if end_idx + 1 < frame_table.num_rows
            else None
        )
        kept_table = set_or_append_column(
            kept_table,
            "frame_index",
            pa.array(range(kept_table.num_rows), type=pa.int64()),
        )
        kept_table = set_or_append_column(
            kept_table,
            "timestamp",
            pa.array(
                (timestamps[start_idx : end_idx + 1] - kept_start_ts).tolist(),
                type=kept_table.column("timestamp").type,
            ),
        )

        trimmed = row.update(frames=frames.with_table(kept_table))
        for video_key, video_ref in row.videos.items():
            base_from_ts = video_ref.from_timestamp_s or 0.0
            trimmed_from_ts = base_from_ts + kept_start_ts
            trimmed_to_ts = (
                trimmed_from_ts + kept_duration_s
                if kept_duration_s is not None
                else video_ref.to_timestamp_s
            )
            trimmed = trimmed.with_video(
                video_key,
                from_timestamp_s=trimmed_from_ts,
                to_timestamp_s=trimmed_to_ts,
            )
            trimmed = trimmed.stats.drop(video_key)

        return trimmed

    return _trim


__all__ = ["motion_trim"]
