from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from refiner.pipeline import LeRobotStatsConfig
from refiner.pipeline.sinks.lerobot._lerobot_frames import compute_episode_stats
from refiner.pipeline.sinks.lerobot._lerobot_stats import (
    _aggregate_stats,
    _feature_stats,
)


def test_feature_stats_computes_expected_quantiles_for_fixed_numeric_fixture() -> None:
    values = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)

    stats = _feature_stats(values)

    np.testing.assert_allclose(stats["min"], np.array([0.0]))
    np.testing.assert_allclose(stats["max"], np.array([4.0]))
    np.testing.assert_allclose(stats["mean"], np.array([2.0]))
    np.testing.assert_allclose(stats["std"], np.array([np.sqrt(2.0)]))
    np.testing.assert_array_equal(stats["count"], np.array([5], dtype=np.int64))
    np.testing.assert_allclose(stats["q01"], np.array([-1e-10]))
    np.testing.assert_allclose(stats["q10"], np.array([-1e-10]))
    np.testing.assert_allclose(stats["q50"], np.array([2.0004000000000204]))
    np.testing.assert_allclose(stats["q90"], np.array([3.99960000009998]))
    np.testing.assert_allclose(stats["q99"], np.array([3.999960000099998]))


def test_compute_episode_stats_frame_quantiles_match_fixed_fixture() -> None:
    frames: list[Mapping[str, Any]] = [
        {"observation.state": [0.0, 10.0]},
        {"observation.state": [1.0, 11.0]},
        {"observation.state": [2.0, 12.0]},
        {"observation.state": [3.0, 13.0]},
        {"observation.state": [4.0, 14.0]},
    ]

    stats = compute_episode_stats(
        frames=frames,
        stats_config=LeRobotStatsConfig(),
    )["observation.state"]

    np.testing.assert_allclose(stats["min"], np.array([0.0, 10.0]))
    np.testing.assert_allclose(stats["max"], np.array([4.0, 14.0]))
    np.testing.assert_allclose(stats["mean"], np.array([2.0, 12.0]))
    np.testing.assert_allclose(
        stats["std"],
        np.array([np.sqrt(2.0), np.sqrt(2.0)]),
    )
    np.testing.assert_array_equal(stats["count"], np.array([5], dtype=np.int64))
    np.testing.assert_allclose(stats["q01"], np.array([-1e-10, 9.9999999999]))
    np.testing.assert_allclose(stats["q10"], np.array([-1e-10, 9.9999999999]))
    np.testing.assert_allclose(
        stats["q50"],
        np.array([2.0004000000000204, 12.00040000000002]),
    )
    np.testing.assert_allclose(
        stats["q90"],
        np.array([3.99960000009998, 13.999600000099981]),
    )
    np.testing.assert_allclose(
        stats["q99"],
        np.array([3.999960000099998, 13.999960000099998]),
    )


def test_aggregate_stats_computes_expected_weighted_quantiles() -> None:
    stats_list = [
        {
            "feat": {
                "min": np.array([0.0]),
                "max": np.array([2.0]),
                "mean": np.array([1.0]),
                "std": np.array([0.5]),
                "count": np.array([3]),
                "q01": np.array([0.1]),
                "q10": np.array([0.2]),
                "q50": np.array([1.0]),
                "q90": np.array([1.8]),
                "q99": np.array([1.98]),
            }
        },
        {
            "feat": {
                "min": np.array([10.0]),
                "max": np.array([12.0]),
                "mean": np.array([11.0]),
                "std": np.array([0.25]),
                "count": np.array([2]),
                "q01": np.array([10.1]),
                "q10": np.array([10.2]),
                "q50": np.array([11.0]),
                "q90": np.array([11.8]),
                "q99": np.array([11.98]),
            }
        },
    ]

    stats = _aggregate_stats(stats_list)["feat"]

    np.testing.assert_allclose(stats["min"], np.array([0.0]))
    np.testing.assert_allclose(stats["max"], np.array([12.0]))
    np.testing.assert_allclose(stats["mean"], np.array([5.0]))
    np.testing.assert_allclose(stats["std"], np.array([4.916807907575809]))
    np.testing.assert_allclose(stats["count"], np.array([5.0]))
    np.testing.assert_allclose(stats["q01"], np.array([4.1]))
    np.testing.assert_allclose(stats["q10"], np.array([4.2]))
    np.testing.assert_allclose(stats["q50"], np.array([5.0]))
    np.testing.assert_allclose(stats["q90"], np.array([5.8]))
    np.testing.assert_allclose(stats["q99"], np.array([5.98]))
