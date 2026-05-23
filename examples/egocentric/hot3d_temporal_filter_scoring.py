from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from refiner.robotics.egocentric.benchmark import score_vla_relative_actions
from refiner.robotics.egocentric.hawor import load_hawor_result_file
from refiner.robotics.egocentric.hot3d import load_hot3d_tar_ground_truth
from refiner.robotics.egocentric.types import HandSide, HaworResult, as_transform_series


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_SUMMARY = ROOT / "artifacts/hot3d-focal-clean-10-summary.json"
DEFAULT_OUTPUT_PREFIX = ROOT / "artifacts/hot3d-temporal-filter-scoring"
HAND_SIDES: tuple[HandSide, ...] = ("left", "right")


@dataclass(frozen=True)
class PredictionRecord:
    clip: str
    experiment: str
    prediction_path: Path
    ground_truth_path: Path
    stream_id: str = "214-1"
    fps: float = 30.0


@dataclass(frozen=True)
class FilterSpec:
    name: str
    fn: Callable[[HaworResult], HaworResult]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score temporal filters on saved HOT3D HaWoR outputs. The filters only "
            "read predictions; HOT3D ground truth is loaded only for final scoring."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Root containing raw HaWoR JSON outputs. Defaults to run_root from "
            "artifacts/hot3d-focal-clean-10-summary.json when present."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional JSONL manifest with clip, experiment, prediction_path, and "
            "ground_truth_path fields. This is the most reliable mode."
        ),
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        help="Directory of HOT3D annotation tar files, used when manifest rows omit GT.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment-name allowlist.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    source_root = args.source_root or _default_source_root()
    records = _load_records(
        source_root=source_root,
        manifest_path=args.manifest,
        ground_truth_root=args.ground_truth_root,
        experiment_allowlist=set(args.experiments) if args.experiments else None,
    )
    if args.limit is not None:
        records = records[: args.limit]
    scores_path = args.output_prefix.with_name(
        args.output_prefix.name + "-scores.jsonl"
    )
    summary_path = args.output_prefix.with_name(
        args.output_prefix.name + "-summary.json"
    )
    logbook_path = args.output_prefix.with_name(args.output_prefix.name + "-logbook.md")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        searched = [str(source_root)]
        if args.manifest:
            searched.append(str(args.manifest))
        summary = _summarize(
            source_root=source_root,
            records=[],
            scores=[],
            failures=[
                {
                    "clip": "*",
                    "experiment": "*",
                    "error": (
                        "No raw HOT3D HaWoR prediction records found. Pass "
                        "--manifest with prediction_path and ground_truth_path "
                        "fields, or restore the source run_root."
                    ),
                }
            ],
        )
        summary["searched"] = searched
        scores_path.write_text("", encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
        _write_logbook(logbook_path, summary, scores_path, summary_path)
        raise SystemExit(
            "No raw HOT3D HaWoR prediction records found. Pass --manifest with "
            "prediction_path and ground_truth_path fields, or restore the run_root "
            f"referenced by {DEFAULT_SOURCE_SUMMARY}. Searched: {searched}"
        )

    filters = _filter_specs()

    all_scores: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    gt_cache: dict[tuple[Path, str, float], HaworResult] = {}
    with scores_path.open("w", encoding="utf-8") as raw_scores:
        for record in records:
            try:
                prediction = load_hawor_result_file(record.prediction_path)
                target = _load_target(record, gt_cache)
            except Exception as exc:  # noqa: BLE001 - experiment log should keep going.
                failures.append(
                    {
                        "clip": record.clip,
                        "experiment": record.experiment,
                        "error": str(exc),
                    }
                )
                continue

            for spec in filters:
                try:
                    filtered = spec.fn(prediction)
                    for horizon in args.horizons:
                        score = score_vla_relative_actions(
                            filtered,
                            target,
                            horizon=horizon,
                            confidence_threshold=args.confidence_threshold,
                        )
                        row = {
                            "clip": record.clip,
                            "source_experiment": record.experiment,
                            "filter": spec.name,
                            "experiment": f"{record.experiment}__{spec.name}",
                            "horizon": horizon,
                            "score": score["score"],
                            "aggregate": score["aggregate"],
                            "hands": score["hands"],
                        }
                        raw_scores.write(json.dumps(row, allow_nan=True) + "\n")
                        all_scores.append(row)
                except Exception as exc:  # noqa: BLE001 - keep independent filters isolated.
                    failures.append(
                        {
                            "clip": record.clip,
                            "experiment": f"{record.experiment}__{spec.name}",
                            "error": str(exc),
                        }
                    )

    summary = _summarize(
        source_root=source_root,
        records=records,
        scores=all_scores,
        failures=failures,
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_logbook(logbook_path, summary, scores_path, summary_path)
    print(json.dumps({"summary": str(summary_path), "scores": str(scores_path)}))


def _default_source_root() -> Path:
    if DEFAULT_SOURCE_SUMMARY.exists():
        payload = json.loads(DEFAULT_SOURCE_SUMMARY.read_text(encoding="utf-8"))
        run_root = payload.get("run_root")
        if run_root:
            return Path(str(run_root))
    return ROOT / "artifacts"


def _load_records(
    *,
    source_root: Path,
    manifest_path: Path | None,
    ground_truth_root: Path | None,
    experiment_allowlist: set[str] | None,
) -> list[PredictionRecord]:
    if manifest_path is not None:
        records = _records_from_manifest(manifest_path, ground_truth_root)
    else:
        records = _discover_records(source_root, ground_truth_root)
    if experiment_allowlist is None:
        return records
    return [record for record in records if record.experiment in experiment_allowlist]


def _records_from_manifest(
    manifest_path: Path, ground_truth_root: Path | None
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    with manifest_path.open("r", encoding="utf-8") as raw:
        for line in raw:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction_path = Path(
                row.get("prediction_path")
                or row.get("hawor_path")
                or row.get("hawor_result_path")
                or row.get("result_path")
            )
            clip = str(row.get("clip") or prediction_path.parent.name)
            experiment = str(
                row.get("experiment") or prediction_path.parent.parent.name
            )
            gt_value = row.get("ground_truth_path") or row.get("gt_path")
            ground_truth_path = (
                Path(gt_value)
                if gt_value
                else _find_ground_truth(ground_truth_root, clip)
            )
            records.append(
                PredictionRecord(
                    clip=clip,
                    experiment=experiment,
                    prediction_path=prediction_path,
                    ground_truth_path=ground_truth_path,
                    stream_id=str(row.get("stream_id", "214-1")),
                    fps=float(row.get("fps", 30.0)),
                )
            )
    return records


def _discover_records(
    source_root: Path, ground_truth_root: Path | None
) -> list[PredictionRecord]:
    if not source_root.exists():
        return []
    records: list[PredictionRecord] = []
    for path in sorted(source_root.rglob("*.json")):
        if path.name in {"summary.json", "workers.json", "manifest.json"}:
            continue
        payload = _read_json_object(path)
        if payload is None or not _looks_like_hawor_result(payload):
            continue
        clip = str(
            (payload.get("metadata") or {}).get("clip")
            or _first_clip_token(path)
            or path.parent.name
        )
        experiment = str(
            (payload.get("metadata") or {}).get("experiment") or path.parent.parent.name
        )
        try:
            gt_path = _find_ground_truth(ground_truth_root or source_root, clip)
        except FileNotFoundError:
            continue
        records.append(
            PredictionRecord(
                clip=clip,
                experiment=experiment,
                prediction_path=path,
                ground_truth_path=gt_path,
            )
        )
    return records


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as raw:
            payload = json.load(raw)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _looks_like_hawor_result(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("timestamps"), list)
        and isinstance(payload.get("camera"), dict)
        and (
            isinstance(payload.get("left_hand"), dict)
            or isinstance(payload.get("right_hand"), dict)
        )
    )


def _first_clip_token(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("clip-"):
            return part
    return None


def _find_ground_truth(root: Path | None, clip: str) -> Path:
    if root is None:
        raise FileNotFoundError(f"no ground-truth root for {clip}")
    candidates = [
        root / f"{clip}.tar",
        root / f"{clip}.tgz",
        root / f"{clip}.tar.gz",
        root / clip / "annotations.tar",
        root / clip / f"{clip}.tar",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(f"*{clip}*.tar"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"could not find HOT3D annotation tar for {clip} in {root}")


def _load_target(
    record: PredictionRecord,
    cache: dict[tuple[Path, str, float], HaworResult],
) -> HaworResult:
    key = (record.ground_truth_path, record.stream_id, record.fps)
    if key not in cache:
        cache[key] = load_hot3d_tar_ground_truth(
            record.ground_truth_path,
            stream_id=record.stream_id,
            fps=record.fps,
        )
    return cache[key]


def _filter_specs() -> list[FilterSpec]:
    specs = [
        FilterSpec("identity", lambda result: result),
        FilterSpec(
            "moving_average_w5", lambda result: _filter_result(result, "mean", window=5)
        ),
        FilterSpec(
            "moving_average_w9", lambda result: _filter_result(result, "mean", window=9)
        ),
        FilterSpec(
            "median_w3", lambda result: _filter_result(result, "median", window=3)
        ),
        FilterSpec(
            "median_w5", lambda result: _filter_result(result, "median", window=5)
        ),
        FilterSpec(
            "median_w9", lambda result: _filter_result(result, "median", window=9)
        ),
        FilterSpec(
            "local_poly_w5_p2",
            lambda result: _filter_result(result, "poly", window=5, order=2),
        ),
        FilterSpec(
            "local_poly_w9_p2",
            lambda result: _filter_result(result, "poly", window=9, order=2),
        ),
        FilterSpec(
            "one_euro_c1.0_b0.01",
            lambda result: _filter_result(
                result, "one_euro", min_cutoff=1.0, beta=0.01
            ),
        ),
        FilterSpec(
            "one_euro_c1.0_b0.05",
            lambda result: _filter_result(
                result, "one_euro", min_cutoff=1.0, beta=0.05
            ),
        ),
        FilterSpec(
            "velocity_clip_0.5mps",
            lambda result: _filter_result(result, "velocity_clip", max_speed=0.5),
        ),
        FilterSpec(
            "velocity_clip_1.0mps",
            lambda result: _filter_result(result, "velocity_clip", max_speed=1.0),
        ),
        FilterSpec(
            "confidence_interp_t0.25",
            lambda result: _filter_result(
                result, "confidence_interp", confidence_threshold=0.25
            ),
        ),
        FilterSpec(
            "confidence_interp_t0.5",
            lambda result: _filter_result(
                result, "confidence_interp", confidence_threshold=0.5
            ),
        ),
        FilterSpec(
            "confidence_interp_t0.25_then_one_euro",
            lambda result: _filter_result(
                _filter_result(result, "confidence_interp", confidence_threshold=0.25),
                "one_euro",
                min_cutoff=1.0,
                beta=0.01,
            ),
        ),
        FilterSpec(
            "confidence_interp_t0.25_then_poly_w5",
            lambda result: _filter_result(
                _filter_result(result, "confidence_interp", confidence_threshold=0.25),
                "poly",
                window=5,
                order=2,
            ),
        ),
    ]
    return specs


def _filter_result(result: HaworResult, kind: str, **params: Any) -> HaworResult:
    payload = copy.deepcopy(result.to_dict())
    timestamps = np.asarray(result.timestamps, dtype=np.float64)
    for side in HAND_SIDES:
        hand = payload.get(f"{side}_hand")
        if not isinstance(hand, dict):
            continue
        confidence = _confidence(hand, len(timestamps))
        if "T_world_wrist" in hand:
            transforms = as_transform_series(
                hand["T_world_wrist"],
                name=f"{side}_hand.T_world_wrist",
            ).copy()
            translations = transforms[:, :3, 3]
            transforms[:, :3, 3] = _filter_series(
                translations,
                timestamps=timestamps,
                confidence=confidence,
                kind=kind,
                **params,
            )
            hand["T_world_wrist"] = transforms.tolist()
        if "mano_pose" in hand:
            pose = np.asarray(hand["mano_pose"], dtype=np.float64)
            if pose.ndim == 2 and len(pose) == len(timestamps):
                hand["mano_pose"] = _filter_series(
                    pose,
                    timestamps=timestamps,
                    confidence=confidence,
                    kind=kind,
                    **params,
                ).tolist()
    metadata = dict(payload.get("metadata") or {})
    metadata["temporal_filter"] = {"kind": kind, **params}
    metadata["filter_uses_ground_truth"] = False
    payload["metadata"] = metadata
    return HaworResult.from_mapping(payload)


def _confidence(hand: dict[str, Any], expected: int) -> np.ndarray | None:
    if "confidence" not in hand:
        return None
    confidence = np.asarray(hand["confidence"], dtype=np.float64)
    if confidence.ndim != 1 or len(confidence) != expected:
        return None
    return confidence


def _filter_series(
    values: np.ndarray,
    *,
    timestamps: np.ndarray,
    confidence: np.ndarray | None,
    kind: str,
    **params: Any,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    base = _interpolate_nonfinite(values)
    if kind == "confidence_interp":
        threshold = float(params["confidence_threshold"])
        return _interpolate_low_confidence(base, confidence, threshold)
    if kind == "mean":
        return _rolling_reduce(base, int(params["window"]), np.nanmean)
    if kind == "median":
        return _rolling_reduce(base, int(params["window"]), np.nanmedian)
    if kind == "poly":
        return _local_polynomial(base, int(params["window"]), int(params["order"]))
    if kind == "one_euro":
        return _one_euro(
            base,
            timestamps=timestamps,
            min_cutoff=float(params["min_cutoff"]),
            beta=float(params["beta"]),
        )
    if kind == "velocity_clip":
        return _velocity_clip(
            base,
            timestamps=timestamps,
            max_speed=float(params["max_speed"]),
        )
    raise ValueError(f"unknown filter kind: {kind}")


def _interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    return _interpolate_masked(values, finite)


def _interpolate_low_confidence(
    values: np.ndarray, confidence: np.ndarray | None, threshold: float
) -> np.ndarray:
    if confidence is None:
        return values.copy()
    valid = (
        np.isfinite(values).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= threshold)
    )
    return _interpolate_masked(values, valid)


def _interpolate_masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = values.copy()
    x = np.arange(len(values), dtype=np.float64)
    if valid.sum() == 0:
        return out
    for col in range(values.shape[1]):
        series = values[:, col]
        out[:, col] = np.interp(x, x[valid], series[valid])
    return out


def _rolling_reduce(
    values: np.ndarray, window: int, reducer: Callable[[np.ndarray, int], np.ndarray]
) -> np.ndarray:
    window = _odd_window(window)
    radius = window // 2
    out = np.empty_like(values)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        stop = min(len(values), idx + radius + 1)
        out[idx] = reducer(values[start:stop], axis=0)
    return out


def _local_polynomial(values: np.ndarray, window: int, order: int) -> np.ndarray:
    window = _odd_window(window)
    if len(values) < 3:
        return values.copy()
    try:
        from scipy.signal import savgol_filter  # type: ignore[import-not-found]

        scipy_window = min(window, len(values) if len(values) % 2 else len(values) - 1)
        if scipy_window >= order + 2:
            return savgol_filter(
                values,
                window_length=scipy_window,
                polyorder=min(order, scipy_window - 1),
                axis=0,
                mode="interp",
            )
    except Exception:  # noqa: BLE001 - numpy fallback is deterministic.
        pass

    radius = window // 2
    out = np.empty_like(values)
    x_all = np.arange(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        stop = min(len(values), idx + radius + 1)
        x = x_all[start:stop] - float(idx)
        fit_order = min(order, len(x) - 1)
        for col in range(values.shape[1]):
            coeff = np.polyfit(x, values[start:stop, col], fit_order)
            out[idx, col] = np.polyval(coeff, 0.0)
    return out


def _one_euro(
    values: np.ndarray,
    *,
    timestamps: np.ndarray,
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float = 1.0,
) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0]
    dx_hat = np.zeros(values.shape[1], dtype=np.float64)
    prev = values[0].copy()
    for idx in range(1, len(values)):
        dt = max(float(timestamps[idx] - timestamps[idx - 1]), 1e-6)
        dx = (values[idx] - prev) / dt
        alpha_d = _one_euro_alpha(dt, derivative_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * dx_hat
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        alpha = _one_euro_alpha(dt, cutoff)
        out[idx] = alpha * values[idx] + (1.0 - alpha) * out[idx - 1]
        prev = values[idx]
    return out


def _one_euro_alpha(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _velocity_clip(
    values: np.ndarray,
    *,
    timestamps: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    out = values.copy()
    for idx in range(1, len(values)):
        dt = max(float(timestamps[idx] - timestamps[idx - 1]), 1e-6)
        delta = values[idx] - out[idx - 1]
        norm = np.linalg.norm(delta)
        max_delta = max_speed * dt
        if norm > max_delta > 0.0:
            out[idx] = out[idx - 1] + delta * (max_delta / norm)
    return out


def _odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("window must be positive")
    return window if window % 2 else window + 1


def _summarize(
    *,
    source_root: Path,
    records: list[PredictionRecord],
    scores: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        if row["score"] is not None:
            grouped[(str(row["experiment"]), int(row["horizon"]))].append(row)

    experiments: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    for (experiment, horizon), rows in sorted(grouped.items()):
        score_values = [float(row["score"]) for row in rows]
        translations = _mean_metric(rows, "wrist_delta_translation_error_m")
        rotations = _mean_metric(rows, "wrist_delta_rotation_error_deg")
        jitter = _mean_metric(rows, "predicted_action_jitter_m")
        coverage = _mean_field(rows, "coverage")
        stats = {
            "count": len(rows),
            "mean_score": float(statistics.fmean(score_values)),
            "median_score": float(statistics.median(score_values)),
            "mean_translation_m": translations,
            "mean_rotation_deg": rotations,
            "mean_jitter_m": jitter,
            "mean_coverage": coverage,
        }
        experiments.setdefault(experiment, {})[str(horizon)] = stats
        if horizon == 1 and (best is None or stats["mean_score"] < best["mean_score"]):
            best = {"experiment": experiment, "horizon": horizon, **stats}

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "record_count": len(records),
        "score_count": len(scores),
        "failure_count": len(failures),
        "failures": failures[:50],
        "best_horizon1": best,
        "experiments": experiments,
    }


def _mean_metric(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get("aggregate", {}).get(key)
        if value is not None:
            values.append(float(value))
    return float(statistics.fmean(values)) if values else None


def _mean_field(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get("aggregate", {}).get(key)
        if value is not None:
            values.append(float(value))
    return float(statistics.fmean(values)) if values else None


def _write_logbook(
    path: Path,
    summary: dict[str, Any],
    scores_path: Path,
    summary_path: Path,
) -> None:
    best = summary.get("best_horizon1")
    lines = [
        "---",
        "title: HOT3D temporal filter scoring logbook",
        "description: Scoring-only temporal filter experiment over saved HOT3D HaWoR outputs.",
        "---",
        "",
        "# HOT3D Temporal Filter Scoring",
        "",
        f"- Time: `{summary['time']}`",
        f"- Source root: `{summary['source_root']}`",
        f"- Records: `{summary['record_count']}`",
        f"- Scores: `{summary['score_count']}`",
        f"- Failures: `{summary['failure_count']}`",
        f"- Scores JSONL: `{scores_path}`",
        f"- Summary JSON: `{summary_path}`",
        "",
        "Filters are prediction-only. HOT3D ground truth is loaded only after filtering for `score_vla_relative_actions`.",
        "",
    ]
    if best is None:
        lines.extend(["## Best Horizon-1 Score", "", "No successful scores."])
    else:
        lines.extend(
            [
                "## Best Horizon-1 Score",
                "",
                f"- Experiment: `{best['experiment']}`",
                f"- Mean score: `{best['mean_score']:.6f}`",
                f"- Mean translation: `{best.get('mean_translation_m')}` m",
                f"- Mean rotation: `{best.get('mean_rotation_deg')}` deg",
                f"- Mean jitter: `{best.get('mean_jitter_m')}` m",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
