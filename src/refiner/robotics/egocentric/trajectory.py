from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from refiner.robotics.egocentric.types import EgocentricRecording, as_transform_series


@dataclass(frozen=True, slots=True)
class TrajectoryQualityGate:
    """Validate and optionally scale camera trajectories before world projection."""

    reference_camera: dict[str, Any] | None = None
    max_total_translation_m: float = 2.0
    max_frame_jump_m: float = 0.5
    max_reference_scale_factor: float = 3.0
    apply_reference_scale: bool = False
    metadata_key: str = "trajectory_qa"

    def run(self, recording: EgocentricRecording) -> EgocentricRecording:
        if recording.camera is None or "T_world_camera" not in recording.camera:
            raise ValueError("TrajectoryQualityGate requires camera.T_world_camera")

        camera = as_transform_series(
            recording.camera["T_world_camera"],
            name="camera.T_world_camera",
        )
        metrics = trajectory_metrics(camera)
        status = "accepted"
        reasons: list[str] = []
        scale_factor = None

        if metrics["total_translation_m"] > self.max_total_translation_m:
            status = "rejected"
            reasons.append("camera_total_translation_exceeds_limit")
        if metrics["max_frame_jump_m"] > self.max_frame_jump_m:
            status = "rejected"
            reasons.append("camera_frame_jump_exceeds_limit")

        if self.reference_camera is not None:
            reference = as_transform_series(
                self.reference_camera["T_world_camera"],
                name="reference_camera.T_world_camera",
            )
            ref_metrics = trajectory_metrics(reference)
            scale_factor = reference_scale_factor(reference, camera)
            metrics["reference_total_translation_m"] = ref_metrics[
                "total_translation_m"
            ]
            metrics["reference_scale_factor"] = scale_factor
            if scale_factor is not None:
                correction = max(scale_factor, 1.0 / scale_factor)
                metrics["reference_scale_correction_factor"] = correction
                if correction > self.max_reference_scale_factor:
                    status = "rejected"
                    reasons.append("reference_scale_correction_exceeds_limit")
                elif self.apply_reference_scale and status == "accepted":
                    camera = scale_camera_translation(camera, scale_factor)
                    metrics["applied_reference_scale_factor"] = scale_factor

        qa = {
            "status": status,
            "reasons": reasons,
            "metrics": metrics,
            "thresholds": {
                "max_total_translation_m": self.max_total_translation_m,
                "max_frame_jump_m": self.max_frame_jump_m,
                "max_reference_scale_factor": self.max_reference_scale_factor,
            },
        }
        metadata = dict(recording.metadata or {})
        metadata[self.metadata_key] = qa
        camera_payload = dict(recording.camera)
        camera_payload["T_world_camera"] = camera.tolist()
        camera_payload[self.metadata_key] = qa
        return recording.replace(camera=camera_payload, metadata=metadata)


def trajectory_metrics(transforms: Any) -> dict[str, Any]:
    camera = as_transform_series(transforms, name="trajectory")
    positions = camera[:, :3, 3]
    frame_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_translation = float(np.linalg.norm(positions[-1] - positions[0]))
    path_length = float(frame_steps.sum()) if len(frame_steps) else 0.0
    return {
        "total_translation_m": total_translation,
        "path_length_m": path_length,
        "max_frame_jump_m": float(frame_steps.max()) if len(frame_steps) else 0.0,
        "median_frame_jump_m": float(np.median(frame_steps))
        if len(frame_steps)
        else 0.0,
        "p95_frame_jump_m": float(np.percentile(frame_steps, 95))
        if len(frame_steps)
        else 0.0,
        "x_range_m": float(np.ptp(positions[:, 0])),
        "y_range_m": float(np.ptp(positions[:, 1])),
        "z_range_m": float(np.ptp(positions[:, 2])),
    }


def reference_scale_factor(
    reference_transforms: np.ndarray,
    candidate_transforms: np.ndarray,
) -> float | None:
    reference = as_transform_series(reference_transforms, name="reference")
    candidate = as_transform_series(candidate_transforms, name="candidate")
    count = min(len(reference), len(candidate))
    if count < 2:
        return None
    reference_steps = np.linalg.norm(
        np.diff(reference[:count, :3, 3], axis=0),
        axis=1,
    )
    candidate_steps = np.linalg.norm(
        np.diff(candidate[:count, :3, 3], axis=0),
        axis=1,
    )
    valid = candidate_steps > 1e-8
    if not np.any(valid):
        return None
    return float(np.median(reference_steps[valid] / candidate_steps[valid]))


def scale_camera_translation(transforms: np.ndarray, scale: float) -> np.ndarray:
    camera = np.array(as_transform_series(transforms, name="trajectory"), copy=True)
    origin = camera[0, :3, 3].copy()
    camera[:, :3, 3] = origin + (camera[:, :3, 3] - origin) * scale
    return camera


__all__ = [
    "TrajectoryQualityGate",
    "reference_scale_factor",
    "scale_camera_translation",
    "trajectory_metrics",
]
