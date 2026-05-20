from __future__ import annotations

from typing import Any

import numpy as np


def coerce_frame_array(frames: Any) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim == 4 and array.dtype != object:
        return validate_frame_array(array)

    try:
        array = np.stack([np.asarray(frame) for frame in frames])
    except TypeError as exc:
        raise TypeError("frames must be a [T, H, W, C] array or iterable") from exc
    except ValueError as exc:
        raise ValueError("frames must have a stable [H, W, C] shape") from exc
    return validate_frame_array(array)


def validate_frame_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError("frames must have shape [T, H, W, C]")
    if array.shape[0] <= 0:
        raise ValueError("frames must contain at least one frame")
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError("frames must have non-empty height and width")
    if array.shape[3] not in {1, 3, 4}:
        raise ValueError("frames must have 1, 3, or 4 channels")
    return np.ascontiguousarray(array)


def rgb24_frame_array(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("video frames must have shape [H, W, C]")
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    elif frame.shape[2] != 3:
        raise ValueError("video frames must have 1, 3, or 4 channels")

    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)

    if np.issubdtype(frame.dtype, np.floating):
        finite = frame[np.isfinite(frame)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            frame = frame * 255.0
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


__all__ = [
    "coerce_frame_array",
    "rgb24_frame_array",
    "validate_frame_array",
]
