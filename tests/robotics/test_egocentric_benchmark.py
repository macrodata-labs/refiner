from __future__ import annotations

import math

import pytest

from refiner.robotics.egocentric import score_vla_relative_actions


def _identity_with_x(x: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _z_rotation_with_x(theta: float, x: float) -> list[list[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c, -s, 0.0, x],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _payload(
    *,
    xs: list[float],
    rotations: list[float] | None = None,
    confidence: list[float] | None = None,
    mano_pose: list[float] | None = None,
) -> dict:
    rotations = rotations or [0.0 for _ in xs]
    return {
        "timestamps": [0.1 * index for index in range(len(xs))],
        "camera": {"T_world_camera": [_identity_with_x(0.0) for _ in xs]},
        "right_hand": {
            "T_world_wrist": [
                _z_rotation_with_x(theta, x) for theta, x in zip(rotations, xs)
            ],
            "mano_pose": [[value] for value in (mano_pose or xs)],
            "confidence": confidence or [1.0 for _ in xs],
        },
    }


def test_score_vla_relative_actions_prioritizes_delta_translation() -> None:
    target = _payload(xs=[0.0, 0.1, 0.2])
    predicted = _payload(xs=[0.0, 0.12, 0.25])

    score = score_vla_relative_actions(predicted, target, hands=("right",))

    right = score["hands"]["right"]
    assert right["valid_actions"] == 2
    assert right["coverage"] == pytest.approx(1.0)
    assert right["wrist_delta_translation_error_m"]["mean"] == pytest.approx(0.025)
    assert right["wrist_absolute_translation_error_m"]["mean"] == pytest.approx(
        (0.0 + 0.02 + 0.05) / 3
    )
    assert score["aggregate"]["wrist_delta_translation_error_m"] == pytest.approx(0.025)


def test_score_vla_relative_actions_scores_delta_rotation() -> None:
    target = _payload(xs=[0.0, 0.0], rotations=[0.0, 0.0])
    predicted = _payload(xs=[0.0, 0.0], rotations=[0.0, math.radians(10.0)])

    score = score_vla_relative_actions(predicted, target, hands=("right",))

    right = score["hands"]["right"]
    assert right["wrist_delta_rotation_error_deg"]["mean"] == pytest.approx(10.0)


def test_score_vla_relative_actions_applies_confidence_coverage() -> None:
    target = _payload(xs=[0.0, 0.1, 0.2])
    predicted = _payload(xs=[0.0, 0.1, 0.4], confidence=[1.0, 0.1, 1.0])

    score = score_vla_relative_actions(
        predicted,
        target,
        hands=("right",),
        confidence_threshold=0.5,
    )

    right = score["hands"]["right"]
    assert right["valid_actions"] == 1
    assert right["total_actions"] == 2
    assert right["coverage"] == pytest.approx(0.5)
    assert right["wrist_delta_translation_error_m"]["mean"] == pytest.approx(0.0)


def test_score_vla_relative_actions_rejects_mismatched_timestamps() -> None:
    target = _payload(xs=[0.0, 0.1, 0.2])
    predicted = _payload(xs=[0.0, 0.1])

    with pytest.raises(ValueError, match="timestamps"):
        score_vla_relative_actions(predicted, target, hands=("right",))
