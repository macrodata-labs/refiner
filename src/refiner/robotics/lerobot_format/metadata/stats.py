from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

StatValue: TypeAlias = list["StatValue"] | bool | int | float


@dataclass(frozen=True, slots=True)
class LeRobotFeatureStats:
    min: StatValue | None = None
    max: StatValue | None = None
    mean: StatValue | None = None
    std: StatValue | None = None
    count: StatValue | None = None


@dataclass(frozen=True, slots=True)
class LeRobotStatsFile(Mapping[str, LeRobotFeatureStats]):
    features: Mapping[str, LeRobotFeatureStats]

    def __getitem__(self, key: str) -> LeRobotFeatureStats:
        return self.features[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)


def compute_feature_stats(array: np.ndarray) -> LeRobotFeatureStats:
    if array.ndim == 0:
        array = array.reshape(1)
    count = int(array.shape[0])
    return LeRobotFeatureStats(
        min=np.atleast_1d(np.min(array, axis=0)).tolist(),
        max=np.atleast_1d(np.max(array, axis=0)).tolist(),
        mean=np.atleast_1d(np.mean(array, axis=0)).tolist(),
        std=np.atleast_1d(np.std(array, axis=0)).tolist(),
        count=np.array([count], dtype=np.int64).tolist(),
    )


def parse_stats_json(
    payload: Mapping[str, Mapping[str, StatValue | None]],
) -> LeRobotStatsFile:
    return LeRobotStatsFile(
        {
            feature: LeRobotFeatureStats(
                min=stats.get("min"),
                max=stats.get("max"),
                mean=stats.get("mean"),
                std=stats.get("std"),
                count=stats.get("count"),
            )
            for feature, stats in payload.items()
            if isinstance(feature, str) and isinstance(stats, Mapping)
        }
    )


def serialize_stats_json(
    stats: LeRobotStatsFile,
) -> dict[str, dict[str, StatValue | None]]:
    return {
        feature: {
            "min": feature_stats.min,
            "max": feature_stats.max,
            "mean": feature_stats.mean,
            "std": feature_stats.std,
            "count": feature_stats.count,
        }
        for feature, feature_stats in stats.items()
    }


__all__ = [
    "LeRobotFeatureStats",
    "LeRobotStatsFile",
    "StatValue",
    "compute_feature_stats",
    "parse_stats_json",
    "serialize_stats_json",
]
