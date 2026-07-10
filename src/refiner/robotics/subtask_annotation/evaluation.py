from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


class SubtaskSegmentationMetrics(BaseModel):
    r_at_50: float
    r_at_70: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    boundary_mae_s: float | None
    overseg_ratio: float | None
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    creates: int
    deletes: int
    drags: int
    edit_cost_per_min: float


def evaluate_subtask_segments(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    video_duration_s: float,
    iou_threshold: float = 0.5,
    boundary_tolerance_s: float = 0.5,
    drag_tolerance_s: float = 0.1,
) -> SubtaskSegmentationMetrics:
    """Score one video with Hungarian IoU and annotator-centric metrics.

    ``creates`` are unmatched reference segments, ``deletes`` are unmatched
    predictions, and ``drags`` count start/end boundaries from matched segments
    that miss ``drag_tolerance_s``. Edit cost is
    ``(3 * creates + deletes + drags) / video_minutes``.
    """

    duration = float(video_duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video_duration_s must be finite and > 0")
    for value, name in (
        (iou_threshold, "iou_threshold"),
        (boundary_tolerance_s, "boundary_tolerance_s"),
        (drag_tolerance_s, "drag_tolerance_s"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    if iou_threshold > 1:
        raise ValueError("iou_threshold must be <= 1")

    predicted_intervals = _intervals(predicted, name="predicted")
    reference_intervals = _intervals(reference, name="reference")
    weights = [
        [_interval_iou(prediction, target) for target in reference_intervals]
        for prediction in predicted_intervals
    ]
    pairs = _hungarian_maximize(weights)
    matched_at_50 = [
        (left, right) for left, right in pairs if weights[left][right] >= 0.5
    ]
    matched_at_70 = [
        (left, right) for left, right in pairs if weights[left][right] >= 0.7
    ]
    matched = [
        (left, right) for left, right in pairs if weights[left][right] >= iou_threshold
    ]

    precision = _ratio(len(matched), len(predicted_intervals))
    recall = _ratio(len(matched), len(reference_intervals))
    f1 = _f1(precision, recall)
    pair_ious = [weights[left][right] for left, right in pairs]
    mean_iou = (
        sum(pair_ious) / len(pair_ious)
        if pair_ious
        else _empty_score(
            predicted_intervals,
            reference_intervals,
        )
    )

    boundary_errors = [
        error
        for left, right in matched
        for error in (
            abs(predicted_intervals[left][0] - reference_intervals[right][0]),
            abs(predicted_intervals[left][1] - reference_intervals[right][1]),
        )
    ]
    boundary_mae = (
        sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
    )
    creates = len(reference_intervals) - len(matched)
    deletes = len(predicted_intervals) - len(matched)
    drags = sum(error > drag_tolerance_s for error in boundary_errors)
    edit_cost = (3 * creates + deletes + drags) / (duration / 60.0)

    predicted_boundaries = _internal_boundaries(predicted_intervals, duration)
    reference_boundaries = _internal_boundaries(reference_intervals, duration)
    boundary_matches = _boundary_match_count(
        predicted_boundaries,
        reference_boundaries,
        tolerance_s=boundary_tolerance_s,
    )
    boundary_precision = _ratio(boundary_matches, len(predicted_boundaries))
    boundary_recall = _ratio(boundary_matches, len(reference_boundaries))

    return SubtaskSegmentationMetrics(
        r_at_50=_ratio(len(matched_at_50), len(reference_intervals)),
        r_at_70=_ratio(len(matched_at_70), len(reference_intervals)),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        boundary_mae_s=boundary_mae,
        overseg_ratio=(
            len(predicted_intervals) / len(reference_intervals)
            if reference_intervals
            else (1.0 if not predicted_intervals else None)
        ),
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=_f1(boundary_precision, boundary_recall),
        creates=creates,
        deletes=deletes,
        drags=drags,
        edit_cost_per_min=edit_cost,
    )


def boundary_f1(
    predicted: Sequence[float],
    reference: Sequence[float],
    *,
    tolerance_s: float,
) -> float:
    """Return one-to-one boundary F1 at the supplied time tolerance."""

    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and >= 0")
    predicted_values = _finite_boundaries(predicted, name="predicted")
    reference_values = _finite_boundaries(reference, name="reference")
    matches = _boundary_match_count(
        predicted_values,
        reference_values,
        tolerance_s=tolerance_s,
    )
    return _f1(
        _ratio(matches, len(predicted_values)),
        _ratio(matches, len(reference_values)),
    )


def _intervals(
    segments: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> list[tuple[float, float]]:
    intervals = []
    for index, segment in enumerate(segments):
        try:
            start_sec = float(segment["start_sec"])
            end_sec = float(segment["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}[{index}] must contain numeric start_sec and end_sec"
            ) from exc
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise ValueError(f"{name}[{index}] timestamps must be finite")
        if end_sec <= start_sec:
            raise ValueError(f"{name}[{index}] end_sec must be after start_sec")
        intervals.append((start_sec, end_sec))
    return intervals


def _interval_iou(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _hungarian_maximize(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if any(len(row) != columns for row in weights):
        raise ValueError("Hungarian weight matrix must be rectangular")
    if not rows or not columns:
        return []

    size = max(rows, columns)
    costs = [[1.0] * size for _ in range(size)]
    for row in range(rows):
        for column in range(columns):
            costs[row][column] = 1.0 - float(weights[row][column])

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    column_to_row = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        column_to_row[0] = row
        min_value = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = column_to_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[column_to_row[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if column_to_row[column0] == 0:
                break
        while True:
            column1 = way[column0]
            column_to_row[column0] = column_to_row[column1]
            column0 = column1
            if column0 == 0:
                break

    return [
        (row - 1, column - 1)
        for column, row in enumerate(column_to_row[1:], start=1)
        if 0 < row <= rows and column <= columns
    ]


def _internal_boundaries(
    intervals: Sequence[tuple[float, float]],
    duration_s: float,
) -> list[float]:
    values = {
        boundary
        for interval in intervals
        for boundary in interval
        if boundary > 1e-6 and boundary < duration_s - 1e-6
    }
    return sorted(values)


def _finite_boundaries(values: Sequence[float], *, name: str) -> list[float]:
    normalized = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} boundaries must be finite")
    return normalized


def _boundary_match_count(
    predicted: Sequence[float],
    reference: Sequence[float],
    *,
    tolerance_s: float,
) -> int:
    left = 0
    right = 0
    matches = 0
    while left < len(predicted) and right < len(reference):
        delta = predicted[left] - reference[right]
        if abs(delta) <= tolerance_s:
            matches += 1
            left += 1
            right += 1
        elif delta < 0:
            left += 1
        else:
            right += 1
    return matches


def _ratio(numerator: int, denominator: int) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if numerator == 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def _empty_score(
    predicted: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
) -> float:
    return 1.0 if not predicted and not reference else 0.0


__all__ = [
    "SubtaskSegmentationMetrics",
    "boundary_f1",
    "evaluate_subtask_segments",
]
