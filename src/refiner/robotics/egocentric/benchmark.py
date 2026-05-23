from __future__ import annotations

from typing import Any

import numpy as np

from refiner.robotics.egocentric.actions import relative_actions_from_hawor
from refiner.robotics.egocentric.types import HandSide, HaworResult, as_transform_series


def score_vla_relative_actions(
    predicted: HaworResult | dict[str, Any],
    target: HaworResult | dict[str, Any],
    *,
    horizon: int = 1,
    hands: tuple[HandSide, ...] = ("left", "right"),
    confidence_threshold: float | None = 0.0,
) -> dict[str, Any]:
    """Score egocentric hand annotations for VLA relative-action training.

    The primary metric is error between relative wrist actions:

    ``inverse(T_world_wrist[t]) @ T_world_wrist[t + horizon]``.

    Absolute hand pose metrics are included as diagnostics, but the top-level
    ``score`` is action-first: translation delta error, rotation delta error,
    MANO pose delta error, coverage, and temporal jitter.
    """

    pred = _coerce_result(predicted, name="predicted")
    gt = _coerce_result(target, name="target")
    if pred.timestamps != gt.timestamps:
        raise ValueError("predicted and target timestamps must match exactly")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    pred_actions = relative_actions_from_hawor(
        pred,
        horizon=horizon,
        hands=hands,
        include_mano_targets=True,
    )
    gt_actions = relative_actions_from_hawor(
        gt,
        horizon=horizon,
        hands=hands,
        include_mano_targets=True,
    )
    frame_count = len(pred.timestamps)
    action_count = max(0, frame_count - horizon)

    hand_scores: dict[str, Any] = {}
    for side in hands:
        pred_hand = pred.hand(side)
        gt_hand = gt.hand(side)
        if pred_hand is None or gt_hand is None:
            hand_scores[side] = _empty_hand_score(action_count)
            continue

        hand_scores[side] = _score_hand(
            pred_hand=pred_hand,
            gt_hand=gt_hand,
            pred_action=pred_actions["hands"].get(side, {}),
            gt_action=gt_actions["hands"].get(side, {}),
            action_count=action_count,
            horizon=horizon,
            confidence_threshold=confidence_threshold,
        )

    aggregate = _aggregate_hand_scores(hand_scores)
    return {
        "kind": "vla_relative_action_benchmark",
        "horizon": horizon,
        "frame_count": frame_count,
        "action_count": action_count,
        "score": aggregate["score"],
        "aggregate": aggregate,
        "hands": hand_scores,
    }


def _score_hand(
    *,
    pred_hand: dict[str, Any],
    gt_hand: dict[str, Any],
    pred_action: dict[str, Any],
    gt_action: dict[str, Any],
    action_count: int,
    horizon: int,
    confidence_threshold: float | None,
) -> dict[str, Any]:
    valid = np.ones(action_count, dtype=bool)
    if confidence_threshold is not None and "confidence" in pred_hand:
        confidence = np.asarray(pred_hand["confidence"], dtype=np.float64)
        if len(confidence) < action_count:
            raise ValueError(
                f"predicted confidence has {len(confidence)} entries, "
                f"expected at least {action_count}"
            )
        valid &= confidence[:action_count] >= confidence_threshold

    metrics: dict[str, Any] = {
        "valid_actions": int(valid.sum()),
        "total_actions": int(action_count),
        "coverage": _safe_ratio(int(valid.sum()), action_count),
    }

    if "wrist_delta" in pred_action and "wrist_delta" in gt_action:
        pred_delta = as_transform_series(
            pred_action["wrist_delta"], name="predicted wrist_delta"
        )
        gt_delta = as_transform_series(
            gt_action["wrist_delta"], name="target wrist_delta"
        )
        _require_same_length(pred_delta, gt_delta, "wrist_delta")
        pred_delta = pred_delta[:action_count]
        gt_delta = gt_delta[:action_count]
        finite = _finite_transform_mask(pred_delta) & _finite_transform_mask(gt_delta)
        action_valid = valid & finite
        translation_error = np.linalg.norm(
            pred_delta[:, :3, 3] - gt_delta[:, :3, 3],
            axis=1,
        )
        rotation_error = _rotation_angle_degrees(
            np.matmul(
                np.swapaxes(gt_delta[:, :3, :3], 1, 2),
                pred_delta[:, :3, :3],
            )
        )
        metrics["wrist_delta_translation_error_m"] = _series_stats(
            translation_error, action_valid
        )
        metrics["wrist_delta_rotation_error_deg"] = _series_stats(
            rotation_error, action_valid
        )

    if "T_world_wrist" in pred_hand and "T_world_wrist" in gt_hand:
        pred_wrist = as_transform_series(
            pred_hand["T_world_wrist"], name="predicted T_world_wrist"
        )
        gt_wrist = as_transform_series(
            gt_hand["T_world_wrist"], name="target T_world_wrist"
        )
        _require_same_length(pred_wrist, gt_wrist, "T_world_wrist")
        finite = _finite_transform_mask(pred_wrist) & _finite_transform_mask(gt_wrist)
        wrist_valid = finite
        if confidence_threshold is not None and "confidence" in pred_hand:
            confidence = np.asarray(pred_hand["confidence"], dtype=np.float64)
            wrist_valid &= confidence[: len(wrist_valid)] >= confidence_threshold
        absolute_translation_error = np.linalg.norm(
            pred_wrist[:, :3, 3] - gt_wrist[:, :3, 3],
            axis=1,
        )
        metrics["wrist_absolute_translation_error_m"] = _series_stats(
            absolute_translation_error,
            wrist_valid,
        )

    if "mano_pose" in pred_hand and "mano_pose" in gt_hand:
        pred_pose = np.asarray(pred_hand["mano_pose"], dtype=np.float64)
        gt_pose = np.asarray(gt_hand["mano_pose"], dtype=np.float64)
        _require_same_length(pred_pose, gt_pose, "mano_pose")
        mano_count = min(action_count, len(pred_pose) - horizon, len(gt_pose) - horizon)
        if mano_count > 0:
            pred_delta_pose = (
                pred_pose[horizon : horizon + mano_count] - pred_pose[:mano_count]
            )
            gt_delta_pose = (
                gt_pose[horizon : horizon + mano_count] - gt_pose[:mano_count]
            )
            pose_error = np.linalg.norm(pred_delta_pose - gt_delta_pose, axis=1)
            metrics["mano_pose_delta_l2"] = _series_stats(
                pose_error,
                valid[:mano_count],
            )

    if "wrist_delta" in pred_action:
        pred_delta = as_transform_series(
            pred_action["wrist_delta"], name="predicted wrist_delta"
        )[:action_count]
        metrics["predicted_action_jitter_m"] = _series_stats(
            _translation_jitter(pred_delta),
            valid[1:] if len(valid) > 1 else valid[:0],
        )

    metrics["score"] = _hand_action_score(metrics)
    return metrics


def _coerce_result(value: HaworResult | dict[str, Any], *, name: str) -> HaworResult:
    if isinstance(value, HaworResult):
        return value
    if isinstance(value, dict):
        return HaworResult.from_mapping(value)
    raise TypeError(f"{name} must be a HaworResult or mapping")


def _empty_hand_score(action_count: int) -> dict[str, Any]:
    return {
        "valid_actions": 0,
        "total_actions": int(action_count),
        "coverage": 0.0,
        "score": None,
    }


def _aggregate_hand_scores(hand_scores: dict[str, Any]) -> dict[str, Any]:
    scored = [score for score in hand_scores.values() if score.get("score") is not None]
    if not scored:
        return {
            "score": None,
            "coverage": 0.0,
            "valid_actions": 0,
            "total_actions": sum(int(s["total_actions"]) for s in hand_scores.values()),
        }

    valid = np.asarray([float(score["valid_actions"]) for score in scored])
    weights = (
        valid / valid.sum() if valid.sum() > 0 else np.ones(len(scored)) / len(scored)
    )
    aggregate: dict[str, Any] = {
        "score": float(
            np.sum([score["score"] * weights[i] for i, score in enumerate(scored)])
        ),
        "valid_actions": int(
            sum(int(s["valid_actions"]) for s in hand_scores.values())
        ),
        "total_actions": int(
            sum(int(s["total_actions"]) for s in hand_scores.values())
        ),
    }
    aggregate["coverage"] = _safe_ratio(
        aggregate["valid_actions"], aggregate["total_actions"]
    )
    for key in (
        "wrist_delta_translation_error_m",
        "wrist_delta_rotation_error_deg",
        "mano_pose_delta_l2",
        "predicted_action_jitter_m",
    ):
        values = []
        value_weights = []
        for index, score in enumerate(scored):
            if key in score and score[key]["mean"] is not None:
                values.append(score[key]["mean"])
                value_weights.append(weights[index])
        if values:
            aggregate[key] = float(np.average(values, weights=value_weights))
    return aggregate


def _hand_action_score(metrics: dict[str, Any]) -> float | None:
    if metrics["valid_actions"] == 0:
        return None
    score = 0.0
    if "wrist_delta_translation_error_m" in metrics:
        score += 1000.0 * _mean_or_zero(metrics["wrist_delta_translation_error_m"])
    if "wrist_delta_rotation_error_deg" in metrics:
        score += 5.0 * _mean_or_zero(metrics["wrist_delta_rotation_error_deg"])
    if "mano_pose_delta_l2" in metrics:
        score += 10.0 * _mean_or_zero(metrics["mano_pose_delta_l2"])
    if "predicted_action_jitter_m" in metrics:
        score += 100.0 * _mean_or_zero(metrics["predicted_action_jitter_m"])
    score += 100.0 * (1.0 - metrics["coverage"])
    return float(score)


def _series_stats(
    values: np.ndarray, valid: np.ndarray
) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if len(values) != len(valid):
        raise ValueError(
            f"metric series has {len(values)} values and {len(valid)} masks"
        )
    selected = values[valid & np.isfinite(values)]
    if len(selected) == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(len(selected)),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p95": float(np.percentile(selected, 95)),
    }


def _mean_or_zero(stats: dict[str, float | int | None]) -> float:
    mean = stats["mean"]
    return 0.0 if mean is None else float(mean)


def _rotation_angle_degrees(rotations: np.ndarray) -> np.ndarray:
    trace = np.trace(rotations, axis1=1, axis2=2)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _translation_jitter(transforms: np.ndarray) -> np.ndarray:
    if len(transforms) < 2:
        return np.asarray([], dtype=np.float64)
    translations = transforms[:, :3, 3]
    return np.linalg.norm(np.diff(translations, axis=0), axis=1)


def _finite_transform_mask(transforms: np.ndarray) -> np.ndarray:
    return np.isfinite(transforms).all(axis=(1, 2))


def _require_same_length(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if len(left) != len(right):
        raise ValueError(
            f"predicted and target {name} lengths differ: {len(left)} != {len(right)}"
        )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = ["score_vla_relative_actions"]
