from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from refiner.media import VideoFile
from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin


def _motion_energy(frames: Sequence[Row], key: str) -> np.ndarray:
    values = np.asarray(
        [np.asarray(frame[key], dtype=np.float32) for frame in frames],
        dtype=np.float32,
    )
    energy = np.zeros(len(values), dtype=np.float32)
    if len(values) <= 1:
        return energy

    deltas = np.abs(np.diff(values, axis=0))
    energy[1:] = deltas.mean(axis=1)
    return energy


def motion_trim(
    *,
    action_key: str = "action",
    state_key: str = "observation.state",
    threshold: float = 0.001,
    pad_frames: int = 5,
) -> Callable[[Row], Row]:
    """Return a row mapper that trims LeRobot episodes to the active motion window.

    The trim span is inferred from the earliest action/state activity above the
    threshold and then applied to the episode frame list.
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
        action_key=action_key,
        state_key=state_key,
        threshold=threshold,
        pad_frames=pad_frames,
    )
    def _trim(row: Row) -> Row:
        if any(isinstance(row.get(key), VideoFile) for key in row.keys()):
            raise ValueError("motion_trim does not support top-level video columns.")

        frames = row.get("frames")
        first_frame = frames[0] if isinstance(frames, list) and frames else None
        if (
            not isinstance(frames, list)
            or not frames
            or not isinstance(first_frame, Row)
            or any(
                key not in first_frame for key in ("timestamp", action_key, state_key)
            )
        ):
            raise ValueError(
                "motion_trim requires LeRobot-format rows with non-empty 'frames' "
                f"containing 'timestamp', '{action_key}', and '{state_key}'"
            )

        action_active = np.flatnonzero(_motion_energy(frames, action_key) > threshold)
        state_active = np.flatnonzero(_motion_energy(frames, state_key) > threshold)
        if action_active.size == 0 and state_active.size == 0:
            return row

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
        start_idx = max(0, min(start_candidates) - pad_frames)
        end_idx = min(len(frames) - 1, max(end_candidates) + pad_frames)
        kept_frames = frames[start_idx : end_idx + 1]
        kept_start_ts = float(kept_frames[0]["timestamp"])

        for new_idx, frame in enumerate(kept_frames):
            kept_frames[new_idx] = frame.update(
                {
                    "frame_index": new_idx,
                    "timestamp": float(frame["timestamp"]) - kept_start_ts,
                }
            )

        return row.update({"frames": kept_frames})

    return _trim


__all__ = ["motion_trim"]
