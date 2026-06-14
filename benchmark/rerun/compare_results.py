from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Rerun cloud benchmark summary artifacts."
    )
    parser.add_argument("baseline", type=Path, help="Baseline summary.json")
    parser.add_argument("candidate", type=Path, help="Candidate summary.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of tables.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a benchmark summary object")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a results list")
    return payload


def _completed(results: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [result for result in results if result.get("status") == "completed"]


def _mean(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return statistics.fmean(numbers)


def _group_results(summary: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in summary.get("results", []):
        if not isinstance(result, dict):
            continue
        case = result.get("case")
        if isinstance(case, str):
            grouped[case].append(result)
    return dict(grouped)


def _unique_values(results: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values = []
    for result in results:
        value = result.get(key)
        if value is not None and value not in values:
            values.append(value)
    return values


def _warnings(results: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    values = []
    for result in results:
        value = result.get(key)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    return values


def _stage_key(stage: Mapping[str, Any]) -> str:
    name = stage.get("name")
    if isinstance(name, str) and name:
        return name
    return f"stage-{stage.get('index', '?')}"


def _stage_means(results: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    durations: dict[str, list[float | None]] = defaultdict(list)
    for result in results:
        stages = result.get("stage_results")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            durations[_stage_key(stage)].append(stage.get("duration_s"))
    return {stage: _mean(values) for stage, values in durations.items()}


def _delta(
    baseline: float | None,
    candidate: float | None,
) -> tuple[float | None, float | None]:
    if baseline is None or candidate is None:
        return None, None
    absolute = candidate - baseline
    percent = (absolute / baseline * 100.0) if baseline else None
    return absolute, percent


def _comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_grouped = _group_results(baseline)
    candidate_grouped = _group_results(candidate)
    cases = sorted(set(baseline_grouped) | set(candidate_grouped))
    rows: list[dict[str, Any]] = []
    for case in cases:
        baseline_all = baseline_grouped.get(case, [])
        candidate_all = candidate_grouped.get(case, [])
        baseline_completed = _completed(baseline_all)
        candidate_completed = _completed(candidate_all)
        baseline_wall = _mean(
            result.get("cloud_wall_time_s") for result in baseline_completed
        )
        candidate_wall = _mean(
            result.get("cloud_wall_time_s") for result in candidate_completed
        )
        wall_delta_s, wall_delta_pct = _delta(baseline_wall, candidate_wall)
        baseline_stages = _stage_means(baseline_completed)
        candidate_stages = _stage_means(candidate_completed)
        stage_rows = []
        for stage in sorted(set(baseline_stages) | set(candidate_stages)):
            baseline_stage = baseline_stages.get(stage)
            candidate_stage = candidate_stages.get(stage)
            stage_delta_s, stage_delta_pct = _delta(baseline_stage, candidate_stage)
            stage_rows.append(
                {
                    "stage": stage,
                    "baseline_duration_s": baseline_stage,
                    "candidate_duration_s": candidate_stage,
                    "delta_s": stage_delta_s,
                    "delta_pct": stage_delta_pct,
                }
            )
        rows.append(
            {
                "case": case,
                "baseline_completed": len(baseline_completed),
                "candidate_completed": len(candidate_completed),
                "baseline_total": len(baseline_all),
                "candidate_total": len(candidate_all),
                "baseline_planned_shards": _unique_values(
                    baseline_all, "planned_shards"
                ),
                "candidate_planned_shards": _unique_values(
                    candidate_all, "planned_shards"
                ),
                "baseline_planning_warnings": _warnings(
                    baseline_all, "planning_warning"
                ),
                "candidate_planning_warnings": _warnings(
                    candidate_all, "planning_warning"
                ),
                "baseline_wall_time_s": baseline_wall,
                "candidate_wall_time_s": candidate_wall,
                "delta_s": wall_delta_s,
                "delta_pct": wall_delta_pct,
                "stages": stage_rows,
            }
        )
    return {
        "baseline_run_token": baseline.get("run_token"),
        "candidate_run_token": candidate.get("run_token"),
        "baseline_git_ref": baseline.get("git_ref"),
        "candidate_git_ref": candidate.get("git_ref"),
        "cases": rows,
    }


def _format_number(value: Any, *, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.2f}{suffix}"


def _format_values(values: Sequence[Any]) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def _print_table(rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print(
            "  ".join(
                value.ljust(widths[column_index])
                for column_index, value in enumerate(row)
            )
        )
        if index == 0:
            print("  ".join("-" * width for width in widths))


def _print_human(comparison: Mapping[str, Any]) -> None:
    print(
        f"Baseline:  {comparison.get('baseline_run_token')} {comparison.get('baseline_git_ref')}"
    )
    print(
        f"Candidate: {comparison.get('candidate_run_token')} {comparison.get('candidate_git_ref')}"
    )
    print()
    rows = [
        (
            "case",
            "runs",
            "shards",
            "baseline_s",
            "candidate_s",
            "delta_s",
            "delta_pct",
        )
    ]
    for case in comparison["cases"]:
        rows.append(
            (
                str(case["case"]),
                f"{case['baseline_completed']}/{case['baseline_total']} -> "
                f"{case['candidate_completed']}/{case['candidate_total']}",
                f"{_format_values(case['baseline_planned_shards'])} -> "
                f"{_format_values(case['candidate_planned_shards'])}",
                _format_number(case["baseline_wall_time_s"]),
                _format_number(case["candidate_wall_time_s"]),
                _format_number(case["delta_s"]),
                _format_number(case["delta_pct"], suffix="%"),
            )
        )
    _print_table(rows)

    for case in comparison["cases"]:
        stages = case["stages"]
        if not stages:
            continue
        print()
        print(f"{case['case']} stages")
        stage_rows = [("stage", "baseline_s", "candidate_s", "delta_s", "delta_pct")]
        for stage in stages:
            stage_rows.append(
                (
                    str(stage["stage"]),
                    _format_number(stage["baseline_duration_s"]),
                    _format_number(stage["candidate_duration_s"]),
                    _format_number(stage["delta_s"]),
                    _format_number(stage["delta_pct"], suffix="%"),
                )
            )
        _print_table(stage_rows)

    warning_rows = [("case", "side", "planned_shards", "warning")]
    for case in comparison["cases"]:
        for side in ("baseline", "candidate"):
            for warning in case[f"{side}_planning_warnings"]:
                warning_rows.append(
                    (
                        str(case["case"]),
                        side,
                        _format_values(case[f"{side}_planned_shards"]),
                        warning,
                    )
                )
    if len(warning_rows) > 1:
        print()
        print("Planning warnings")
        _print_table(warning_rows)


def main() -> int:
    args = _parse_args()
    comparison = _comparison(
        _load_summary(args.baseline), _load_summary(args.candidate)
    )
    if args.json:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        _print_human(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
