from __future__ import annotations

import pytest

import refiner as mdr
import refiner.robotics


def _profile(*, domain_id: str = "assembly") -> mdr.robotics.DomainProfile:
    return mdr.robotics.DomainProfile(
        domain_id=domain_id,
        version="1",
        policy=mdr.robotics.SegmentationPolicy(
            policy_id="assembly-actions",
            version="2",
            description="One segment per completed assembly state change.",
            action_taxonomy=("pick", "insert", "fasten", "place"),
            target_segments_per_minute=(8, 12),
        ),
        gold_set="assembly-consensus-v1",
    )


def test_domain_profile_hash_is_stable_and_sensitive() -> None:
    profile = _profile()

    assert len(profile.profile_hash) == 64
    assert profile.profile_hash == _profile().profile_hash
    assert profile.profile_hash != _profile(domain_id="walden").profile_hash
    assert "8-12 segments per video minute" in profile.policy.prompt_section()


def test_segmentation_policy_rejects_invalid_density() -> None:
    with pytest.raises(ValueError, match="positive"):
        mdr.robotics.SegmentationPolicy(
            policy_id="bad",
            version="1",
            description="bad density",
            target_segments_per_minute=(10, 5),
        )


def test_validate_subtask_segments_preserves_semantics_and_reports_repairs() -> None:
    validation = mdr.robotics.validate_subtask_segments(
        [
            {"start_sec": -1.0, "end_sec": 2.0, "subtask": "pick"},
            {"start_sec": 1.5, "end_sec": 7.0, "subtask": "place"},
            {"start_sec": float("nan"), "end_sec": 3.0, "subtask": "bad"},
        ],
        video_duration_s=5.0,
    )

    assert validation.status == "partial"
    assert validation.segments == [
        {"start_sec": 0.0, "end_sec": 2.0, "subtask": "pick"},
        {"start_sec": 1.5, "end_sec": 5.0, "subtask": "place"},
    ]
    assert validation.raw_segments[2]["start_sec"] is None
    assert any(
        "preserved without boundary snapping" in issue for issue in validation.issues
    )


def test_evaluate_subtask_segments_scores_identical_timelines() -> None:
    segments = [
        {"start_sec": 0.0, "end_sec": 5.0},
        {"start_sec": 5.0, "end_sec": 10.0},
    ]

    metrics = mdr.robotics.evaluate_subtask_segments(
        segments,
        segments,
        video_duration_s=10.0,
    )

    assert metrics.r_at_50 == 1.0
    assert metrics.r_at_70 == 1.0
    assert metrics.precision == 1.0
    assert metrics.f1 == 1.0
    assert metrics.mean_iou == 1.0
    assert metrics.boundary_mae_s == 0.0
    assert metrics.boundary_f1 == 1.0
    assert metrics.edit_cost_per_min == 0.0


def test_evaluate_subtask_segments_counts_annotation_edits() -> None:
    predicted = [{"start_sec": 0.2, "end_sec": 4.8}]
    reference = [
        {"start_sec": 0.0, "end_sec": 5.0},
        {"start_sec": 5.0, "end_sec": 10.0},
    ]

    metrics = mdr.robotics.evaluate_subtask_segments(
        predicted,
        reference,
        video_duration_s=10.0,
        drag_tolerance_s=0.1,
    )

    assert metrics.creates == 1
    assert metrics.deletes == 0
    assert metrics.drags == 2
    assert metrics.edit_cost_per_min == pytest.approx(30.0)
    assert metrics.overseg_ratio == 0.5


def test_boundary_f1_uses_one_to_one_matching() -> None:
    assert mdr.robotics.boundary_f1(
        [1.0, 1.1, 4.0],
        [1.05, 3.8],
        tolerance_s=0.25,
    ) == pytest.approx(0.8)
