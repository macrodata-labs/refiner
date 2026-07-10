from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field


SegmentationStatus = Literal["ok", "empty", "partial", "invalid", "blocked"]


class SegmentationProvenance(BaseModel):
    domain_id: str
    profile_version: str
    policy_id: str
    policy_version: str
    profile_hash: str
    config_hash: str
    backend: str
    model: str
    count_prior: int | None = None
    fallback_used: bool = False
    render_config: dict[str, int | float]
    latency_ms: float = Field(ge=0.0)
    usage: dict[str, Any] = Field(default_factory=dict)


class SegmentationResult(BaseModel):
    status: SegmentationStatus
    segments: list[dict[str, Any]]
    raw_segments: list[dict[str, Any]]
    issues: list[str] = Field(default_factory=list)
    provenance: SegmentationProvenance
    block_reason: str | None = None


class TimelineValidation(BaseModel):
    segments: list[dict[str, Any]]
    raw_segments: list[dict[str, Any]]
    issues: list[str]
    status: Literal["ok", "empty", "partial", "invalid"]


def validate_subtask_segments(
    value: Sequence[Mapping[str, Any]],
    *,
    video_duration_s: float | None = None,
) -> TimelineValidation:
    """Apply structural validation without semantic snapping or merging."""

    duration = _duration(video_duration_s)
    normalized: list[dict[str, Any]] = []
    raw_segments: list[dict[str, Any]] = []
    issues: list[str] = []

    for index, item in enumerate(value):
        raw = {
            "start_sec": _json_number(item.get("start_sec")),
            "end_sec": _json_number(item.get("end_sec")),
            "subtask": str(item.get("subtask") or ""),
        }
        raw_segments.append(raw)
        try:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"segment[{index}] dropped: invalid start_sec or end_sec")
            continue
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            issues.append(f"segment[{index}] dropped: timestamps must be finite")
            continue

        repaired_start = max(0.0, start_sec)
        repaired_end = max(0.0, end_sec)
        if repaired_start != start_sec:
            issues.append(f"segment[{index}] start_sec clamped to 0")
        if repaired_end != end_sec:
            issues.append(f"segment[{index}] end_sec clamped to 0")
        if duration is not None:
            if repaired_start > duration:
                repaired_start = duration
                issues.append(f"segment[{index}] start_sec clipped to video duration")
            if repaired_end > duration:
                repaired_end = duration
                issues.append(f"segment[{index}] end_sec clipped to video duration")
        if repaired_end <= repaired_start:
            issues.append(f"segment[{index}] dropped: end_sec must be after start_sec")
            continue

        label = str(item.get("subtask") or "").strip() or f"segment {index}"
        normalized.append(
            {
                "start_sec": round(repaired_start, 3),
                "end_sec": round(repaired_end, 3),
                "subtask": label,
            }
        )

    normalized.sort(key=lambda segment: (segment["start_sec"], segment["end_sec"]))
    previous: Mapping[str, Any] | None = None
    for index, segment in enumerate(normalized):
        if previous is not None and segment["start_sec"] < previous["end_sec"]:
            issues.append(
                f"segment[{index}] overlaps the previous normalized segment; "
                "preserved without boundary snapping"
            )
        previous = segment

    if not value:
        status: Literal["ok", "empty", "partial", "invalid"] = "empty"
    elif not normalized:
        status = "invalid"
    elif issues:
        status = "partial"
    else:
        status = "ok"
    return TimelineValidation(
        segments=normalized,
        raw_segments=raw_segments,
        issues=issues,
        status=status,
    )


def _duration(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("video_duration_s must be finite and > 0")
    return value


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "SegmentationProvenance",
    "SegmentationResult",
    "SegmentationStatus",
    "TimelineValidation",
    "validate_subtask_segments",
]
