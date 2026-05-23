from __future__ import annotations

import pytest
import numpy as np

from refiner.robotics.egocentric import (
    EgocentricRecording,
    TrajectoryQualityGate,
    reference_scale_factor,
    scale_camera_translation,
    trajectory_metrics,
)


def _identity_with_x(x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_trajectory_metrics_reports_motion() -> None:
    metrics = trajectory_metrics(
        np.asarray([_identity_with_x(0.0), _identity_with_x(0.5)])
    )

    assert metrics["total_translation_m"] == pytest.approx(0.5)
    assert metrics["max_frame_jump_m"] == pytest.approx(0.5)
    assert metrics["x_range_m"] == pytest.approx(0.5)


def test_reference_scale_factor_uses_median_step_ratio() -> None:
    reference = [_identity_with_x(0.0), _identity_with_x(0.1), _identity_with_x(0.2)]
    candidate = [_identity_with_x(0.0), _identity_with_x(1.0), _identity_with_x(2.0)]

    assert reference_scale_factor(
        np.asarray(reference), np.asarray(candidate)
    ) == pytest.approx(0.1)


def test_scale_camera_translation_preserves_first_origin() -> None:
    scaled = scale_camera_translation(
        np.asarray([_identity_with_x(1.0), _identity_with_x(3.0)]),
        0.5,
    )

    assert scaled[0, 0, 3] == pytest.approx(1.0)
    assert scaled[1, 0, 3] == pytest.approx(2.0)


def test_quality_gate_rejects_large_reference_scale_correction() -> None:
    recording = EgocentricRecording(
        timestamps=[0.0, 0.1],
        camera={"T_world_camera": [_identity_with_x(0.0), _identity_with_x(5.0)]},
    )
    gate = TrajectoryQualityGate(
        reference_camera={
            "T_world_camera": [_identity_with_x(0.0), _identity_with_x(0.5)]
        },
        max_total_translation_m=10.0,
        max_frame_jump_m=10.0,
        max_reference_scale_factor=3.0,
    )

    updated = gate.run(recording)

    qa = updated.metadata["trajectory_qa"]  # type: ignore[index]
    assert qa["status"] == "rejected"
    assert "reference_scale_correction_exceeds_limit" in qa["reasons"]
