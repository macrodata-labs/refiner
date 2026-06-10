from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "motion_trim": "refiner.robotics.motion",
    "reward_score": "refiner.robotics.reward",
    "TimestampedContactSheet": "refiner.robotics.subtask_annotation",
    "contact_sheet_prompt_manifest": "refiner.robotics.subtask_annotation",
    "subtask_annotation": "refiner.robotics.subtask_annotation",
    "timestamped_contact_sheets": "refiner.robotics.subtask_annotation",
    "track_hands": "refiner.robotics.hand_tracking",
    "RoboticsRow": "refiner.robotics.row",
    "RoboticsTabular": "refiner.robotics.tabular",
    "LeRobotRow": "refiner.robotics.lerobot_format",
    "LeRobotTabular": "refiner.robotics.lerobot_format",
    "LeRobotFeatureInfo": "refiner.robotics.lerobot_format",
    "LeRobotFeatureStats": "refiner.robotics.lerobot_format",
    "LeRobotInfo": "refiner.robotics.lerobot_format",
    "LeRobotMetadata": "refiner.robotics.lerobot_format",
    "LeRobotStatsFile": "refiner.robotics.lerobot_format",
    "LeRobotTasks": "refiner.robotics.lerobot_format",
    "LeRobotVideoInfo": "refiner.robotics.lerobot_format",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
