from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def _non_empty(value: str, *, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _string_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} values must be non-empty")
    return normalized


@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    """Versioned definition of which semantic events become segments."""

    policy_id: str
    version: str
    description: str
    action_taxonomy: tuple[str, ...] = ()
    include_events: tuple[str, ...] = ()
    exclude_events: tuple[str, ...] = ()
    target_segments_per_minute: tuple[float, float] | None = None
    additional_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_id",
            _non_empty(self.policy_id, name="policy_id"),
        )
        object.__setattr__(self, "version", _non_empty(self.version, name="version"))
        object.__setattr__(
            self,
            "description",
            _non_empty(self.description, name="description"),
        )
        for name in (
            "action_taxonomy",
            "include_events",
            "exclude_events",
            "additional_rules",
        ):
            object.__setattr__(
                self,
                name,
                _string_tuple(tuple(getattr(self, name)), name=name),
            )
        density = self.target_segments_per_minute
        if density is not None:
            if len(density) != 2:
                raise ValueError(
                    "target_segments_per_minute must contain exactly two values"
                )
            low, high = (float(density[0]), float(density[1]))
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError("target_segments_per_minute values must be finite")
            if low <= 0 or high < low:
                raise ValueError(
                    "target_segments_per_minute must be a positive (low, high) range"
                )
            object.__setattr__(self, "target_segments_per_minute", (low, high))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_section(self) -> str:
        lines = [
            "Segmentation policy:",
            f"- Policy: {self.policy_id}@{self.version}",
            f"- Definition: {self.description}",
        ]
        if self.action_taxonomy:
            lines.append(f"- Action taxonomy: {', '.join(self.action_taxonomy)}")
        lines.extend(f"- Include: {event}" for event in self.include_events)
        lines.extend(f"- Exclude: {event}" for event in self.exclude_events)
        density = self.target_segments_per_minute
        if density is not None:
            lines.append(
                "- Target granularity: "
                f"{density[0]:g}-{density[1]:g} segments per video minute"
            )
        lines.extend(f"- Rule: {rule}" for rule in self.additional_rules)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """Bind a segmentation policy and evaluation artifacts to one domain."""

    domain_id: str
    version: str
    policy: SegmentationPolicy
    model_artifact: str | None = None
    gold_set: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "domain_id",
            _non_empty(self.domain_id, name="domain_id"),
        )
        object.__setattr__(self, "version", _non_empty(self.version, name="version"))
        if not isinstance(self.policy, SegmentationPolicy):
            raise TypeError("policy must be a SegmentationPolicy")
        for name in ("model_artifact", "gold_set"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_empty(value, name=name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


MANIPULATION_EVENTS_V1 = SegmentationPolicy(
    policy_id="manipulation-events",
    version="1",
    description=(
        "One segment per completed manipulation event that changes the held object "
        "or the observable world state."
    ),
    action_taxonomy=(
        "pick",
        "place",
        "open",
        "close",
        "pour",
        "wipe",
        "tool-use",
    ),
    include_events=(
        "A held object changes, is placed, or is released.",
        "A tool starts or stops changing a surface.",
        "A container, door, or lid opens or closes.",
        "Contents move between containers.",
    ),
    exclude_events=(
        "Idle time and camera motion.",
        "Approach, grasp adjustment, small repositioning, and retreat when the "
        "world state does not change.",
    ),
    additional_rules=(
        "Do not merge distinct events that complete different world states.",
        "Short events are valid when a pick, place, open, close, or release completes.",
    ),
)


__all__ = [
    "DomainProfile",
    "MANIPULATION_EVENTS_V1",
    "SegmentationPolicy",
]
