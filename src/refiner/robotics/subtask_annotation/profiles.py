"""Versioned segmentation profiles used by the current robotics benchmarks."""

from refiner.robotics.subtask_annotation.profile import (
    DomainProfile,
    SegmentationPolicy,
)


WALDEN_POLICY_V1 = SegmentationPolicy(
    policy_id="walden-manipulation-events",
    version="1",
    description=(
        "One segment per completed manipulation event that changes the held object "
        "or an observable object, container, or surface state."
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
        "An object is picked up, placed, transferred, or released.",
        "A door, drawer, lid, or container reaches an opened or closed state.",
        "Contents visibly move between containers.",
        "A tool starts or finishes changing a surface or object state.",
    ),
    exclude_events=(
        "Idle time, camera motion, and navigation without a manipulation event.",
        "Approach, grasp adjustment, small repositioning, and retreat when no "
        "observable state change completes.",
    ),
    additional_rules=(
        "Use the event completion visible in the video, not a fixed time window, "
        "to choose boundaries.",
        "Do not merge adjacent events that produce different completed world states.",
    ),
)


WALDEN_V1 = DomainProfile(
    domain_id="walden",
    version="1",
    policy=WALDEN_POLICY_V1,
    model_artifact="vjepa2-vit-l-tridet-walden-epoch-019",
    gold_set="walden-97-operator-v1",
)


ASSEMBLY_POLICY_V1 = SegmentationPolicy(
    policy_id="assembly-relation-events",
    version="1",
    description=(
        "One segment per completed assembly relation or reversible assembly state "
        "change between parts, tools, fasteners, or connectors."
    ),
    action_taxonomy=(
        "pick",
        "position",
        "insert",
        "remove",
        "fasten",
        "unfasten",
        "connect",
        "disconnect",
        "place",
    ),
    include_events=(
        "A part is picked, placed, inserted, removed, attached, or detached.",
        "A fastener reaches a fastened or unfastened state.",
        "A cable, plug, or connector reaches a connected or disconnected state.",
        "A tool completes a state-changing assembly operation.",
    ),
    exclude_events=(
        "Approach, visual inspection, and idle time.",
        "Fine alignment, regrasping, and temporary contact before the intended "
        "assembly relation completes.",
    ),
    additional_rules=(
        "Split when the completed relation changes, even if the same part or tool "
        "remains in hand.",
        "Keep a failed attempt inside the eventual successful event unless it "
        "creates a distinct persistent state that an annotator would edit separately.",
    ),
)


ASSEMBLY_V1 = DomainProfile(
    domain_id="assembly",
    version="1",
    policy=ASSEMBLY_POLICY_V1,
    gold_set="assembly-40x5-consensus-v1",
)


__all__ = [
    "ASSEMBLY_POLICY_V1",
    "ASSEMBLY_V1",
    "WALDEN_POLICY_V1",
    "WALDEN_V1",
]
