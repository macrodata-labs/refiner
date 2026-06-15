from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_compare_results() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "rerun"
        / "compare_results.py"
    )
    spec = importlib.util.spec_from_file_location("compare_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_results = _load_compare_results()


def test_stage_total_is_derived_from_stage_durations() -> None:
    summary = {
        "results": [
            {
                "case": "rrd-copy",
                "status": "completed",
                "cloud_wall_time_s": 66.07,
                "stage_results": [
                    {"name": "write_rerun_stage_0", "duration_s": 14.88},
                    {"name": "write_rerun_stage_1", "duration_s": 3.65},
                ],
            }
        ]
    }

    assert compare_results._stage_total(summary["results"]) == 18.53


def test_comparison_reports_stage_total_delta() -> None:
    baseline = {
        "run_token": "baseline",
        "git_ref": "base",
        "results": [
            {
                "case": "rrd-copy",
                "status": "completed",
                "cloud_wall_time_s": 61.82,
                "stage_results": [
                    {"name": "write_rerun_stage_0", "duration_s": 20.80},
                    {"name": "write_rerun_stage_1", "duration_s": 2.32},
                ],
            }
        ],
    }
    candidate = {
        "run_token": "candidate",
        "git_ref": "cand",
        "results": [
            {
                "case": "rrd-copy",
                "status": "completed",
                "cloud_wall_time_s": 66.07,
                "stage_results": [
                    {"name": "write_rerun_stage_0", "duration_s": 14.88},
                    {"name": "write_rerun_stage_1", "duration_s": 3.65},
                ],
            }
        ],
    }

    comparison = compare_results._comparison(baseline, candidate)
    case = comparison["cases"][0]

    assert case["baseline_stage_total_s"] == 23.12
    assert case["candidate_stage_total_s"] == 18.53
    assert case["stage_total_delta_s"] == -4.59
