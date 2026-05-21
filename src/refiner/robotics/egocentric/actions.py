from __future__ import annotations

from typing import Any

import numpy as np

from refiner.pipeline.data.row import Row
from refiner.pipeline.planning import describe_builtin
from refiner.robotics.egocentric.types import HandSide, HaworResult, as_transform_series


def make_relative_actions(
    *,
    source_column: str = "hawor",
    output_column: str = "ego_actions",
    horizon: int = 1,
    hands: tuple[HandSide, ...] = ("left", "right"),
    include_mano_targets: bool = True,
) -> Any:
    """Return a row mapper that derives relative wrist and hand action targets.

    The source column must contain a normalized HaWoR result. For each hand with
    ``T_world_wrist`` transforms, this produces action records aligned to
    ``timestamps[:-horizon]``. Wrist deltas are expressed as:

    ``inverse(T_world_wrist[t]) @ T_world_wrist[t + horizon]``.
    """

    if not source_column:
        raise ValueError("source_column cannot be empty")
    if not output_column:
        raise ValueError("output_column cannot be empty")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if not hands:
        raise ValueError("hands cannot be empty")

    @describe_builtin(
        "robotics.egocentric:make_relative_actions",
        source_column=source_column,
        output_column=output_column,
        horizon=horizon,
        hands=hands,
        include_mano_targets=include_mano_targets,
    )
    def _actions(row: Row) -> Row:
        payload = row[source_column]
        if not isinstance(payload, dict):
            raise ValueError(f"{source_column} must contain a HaWoR result mapping")
        result = HaworResult.from_mapping(payload)
        actions = relative_actions_from_hawor(
            result,
            horizon=horizon,
            hands=hands,
            include_mano_targets=include_mano_targets,
        )
        if row.shard_id is not None:
            row.log_throughput(
                "egocentric_actions_emitted",
                len(actions["timestamps"]),
                unit="actions",
            )
        return row.update({output_column: actions})

    return _actions


def relative_actions_from_hawor(
    result: HaworResult,
    *,
    horizon: int = 1,
    hands: tuple[HandSide, ...] = ("left", "right"),
    include_mano_targets: bool = True,
) -> dict[str, Any]:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if len(result.timestamps) <= horizon:
        return {
            "timestamps": [],
            "horizon": horizon,
            "hands": {},
        }

    action_count = len(result.timestamps) - horizon
    output: dict[str, Any] = {
        "timestamps": result.timestamps[:action_count],
        "target_timestamps": result.timestamps[horizon:],
        "horizon": horizon,
        "hands": {},
    }
    hand_actions: dict[str, Any] = output["hands"]

    for side in hands:
        hand = result.hand(side)
        if hand is None or "T_world_wrist" not in hand:
            continue
        wrist = as_transform_series(
            hand["T_world_wrist"],
            name=f"{side}_hand.T_world_wrist",
        )
        deltas = [
            (np.linalg.inv(wrist[idx]) @ wrist[idx + horizon]).tolist()
            for idx in range(action_count)
        ]
        side_payload: dict[str, Any] = {"wrist_delta": deltas}
        if include_mano_targets and "mano_pose" in hand:
            side_payload["mano_target"] = list(hand["mano_pose"][horizon:])
        if "confidence" in hand:
            side_payload["confidence"] = list(hand["confidence"][:action_count])
        hand_actions[side] = side_payload

    return output


__all__ = [
    "make_relative_actions",
    "relative_actions_from_hawor",
]
