from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from refiner.robotics.lerobot_format.metadata.info import LeRobotInfo
from refiner.robotics.lerobot_format.metadata.stats import LeRobotStatsFile
from refiner.robotics.lerobot_format.metadata.tasks import LeRobotTasks, merge_tasks


@dataclass(frozen=True, slots=True)
class LeRobotMetadata:
    info: LeRobotInfo
    stats: LeRobotStatsFile
    tasks: LeRobotTasks


def merge_metadata(
    metadata_by_dataset: Sequence[LeRobotMetadata],
) -> tuple[tuple[LeRobotMetadata, ...], tuple[dict[int, int], ...]]:
    merged_tasks, remaps = merge_tasks(
        [metadata.tasks for metadata in metadata_by_dataset]
    )
    return (
        tuple(
            replace(metadata, tasks=merged_tasks) for metadata in metadata_by_dataset
        ),
        remaps,
    )


__all__ = ["LeRobotMetadata", "merge_metadata"]
