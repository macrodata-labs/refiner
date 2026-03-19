from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa

from refiner.robotics.lerobot_format import (
    LeRobotStatsFile,
    compute_feature_stats,
    compute_table_stats,
)


def _assert_stats_close(
    actual: Mapping[str, Any],
    expected: Mapping[str, np.ndarray],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        actual_value = np.asarray(actual[key])
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(actual_value, value)
        else:
            np.testing.assert_allclose(actual_value, value)


def test_feature_stats_computes_expected_quantiles_for_fixed_numeric_fixture() -> None:
    stats = compute_feature_stats(
        np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    ).to_json_dict()

    _assert_stats_close(
        stats,
        {
            "min": np.array([0.0]),
            "max": np.array([4.0]),
            "mean": np.array([2.0]),
            "std": np.array([np.sqrt(2.0)]),
            "count": np.array([5], dtype=np.int64),
            "q01": np.array([-1e-10]),
            "q10": np.array([-1e-10]),
            "q50": np.array([2.0004000000000204]),
            "q90": np.array([3.99960000009998]),
            "q99": np.array([3.999960000099998]),
        },
    )


def test_compute_episode_stats_frame_quantiles_match_fixed_fixture() -> None:
    frames: list[Mapping[str, Any]] = [
        {"observation.state": [0.0, 10.0]},
        {"observation.state": [1.0, 11.0]},
        {"observation.state": [2.0, 12.0]},
        {"observation.state": [3.0, 13.0]},
        {"observation.state": [4.0, 14.0]},
    ]

    stats = compute_table_stats(pa.Table.from_pylist(list(frames))).to_json_dict()[
        "observation.state"
    ]

    _assert_stats_close(
        stats,
        {
            "min": np.array([0.0, 10.0]),
            "max": np.array([4.0, 14.0]),
            "mean": np.array([2.0, 12.0]),
            "std": np.array([np.sqrt(2.0), np.sqrt(2.0)]),
            "count": np.array([5], dtype=np.int64),
            "q01": np.array([-1e-10, 9.9999999999]),
            "q10": np.array([-1e-10, 9.9999999999]),
            "q50": np.array([2.0004000000000204, 12.00040000000002]),
            "q90": np.array([3.99960000009998, 13.999600000099981]),
            "q99": np.array([3.999960000099998, 13.999960000099998]),
        },
    )


def test_aggregate_stats_computes_expected_weighted_quantiles() -> None:
    stats = LeRobotStatsFile.aggregate(
        [
            LeRobotStatsFile.from_json_dict(
                {
                    "feat": {
                        "min": [0.0],
                        "max": [2.0],
                        "mean": [1.0],
                        "std": [0.5],
                        "count": [3],
                        "q01": [0.1],
                        "q10": [0.2],
                        "q50": [1.0],
                        "q90": [1.8],
                        "q99": [1.98],
                    }
                }
            ),
            LeRobotStatsFile.from_json_dict(
                {
                    "feat": {
                        "min": [10.0],
                        "max": [12.0],
                        "mean": [11.0],
                        "std": [0.25],
                        "count": [2],
                        "q01": [10.1],
                        "q10": [10.2],
                        "q50": [11.0],
                        "q90": [11.8],
                        "q99": [11.98],
                    }
                }
            ),
        ]
    ).to_json_dict()["feat"]

    _assert_stats_close(
        stats,
        {
            "min": np.array([0.0]),
            "max": np.array([12.0]),
            "mean": np.array([5.0]),
            "std": np.array([4.916807907575809]),
            "count": np.array([5.0]),
            "q01": np.array([4.1]),
            "q10": np.array([4.2]),
            "q50": np.array([5.0]),
            "q90": np.array([5.8]),
            "q99": np.array([5.98]),
        },
    )
