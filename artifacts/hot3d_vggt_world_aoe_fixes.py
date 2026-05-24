from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


BASE_SCRIPT = Path("/cache/hot3d/scripts/hot3d_vggt_focal_hawor.py")
STATE_SCRIPT = Path("/cache/hot3d/scripts/hot3d_state_joint_delta_loo.py")
VGGT_WORLD_SCRIPT = Path("/cache/hot3d/scripts/hot3d_vggt_world_final_experiments.py")
AOE_SCRIPT = Path("/cache/hot3d/scripts/hot3d_aoe_vitra_next_experiments.py")
REMAINING_SCRIPT = Path("/cache/hot3d/scripts/hot3d_remaining_experiments.py")

SRC = Path("/cache/hot3d/benchmarks/focal-clean-10/vggt_scaled1408")
RUN = Path("/cache/hot3d/benchmarks/focal-clean-10-vggt-world-aoe-fixes")
LOG = RUN / "logbook.jsonl"

HANDS = ("left", "right")
HORIZONS = (1, 3, 5, 10)


def import_file(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = import_file(BASE_SCRIPT, "base_hot3d")
state = import_file(STATE_SCRIPT, "state_delta")
vggt_world_mod = import_file(VGGT_WORLD_SCRIPT, "vggt_world_final")
aoe = import_file(AOE_SCRIPT, "aoe_next")
remaining = import_file(REMAINING_SCRIPT, "remaining_experiments")


def log(event: str, **kwargs: Any) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **kwargs,
    }
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=True) + "\n")
    print(json.dumps(row, allow_nan=True), flush=True)


def copy_payload(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=True))


def summarize(rows: list[dict[str, Any]], clips: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_root": str(RUN),
        "clip_count": len(clips),
        "experiments": {},
    }
    for experiment in sorted({row["experiment"] for row in rows}):
        summary["experiments"][experiment] = {}
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment
                and row["horizon"] == horizon
                and row.get("score") is not None
            ]
            if not selected:
                continue
            summary["experiments"][experiment][str(horizon)] = {
                "count": len(selected),
                "mean_score": float(np.mean([row["score"] for row in selected])),
                "median_score": float(np.median([row["score"] for row in selected])),
                "mean_translation_m": float(
                    np.nanmean(
                        [
                            row["aggregate"].get(
                                "wrist_delta_translation_error_m", math.nan
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_rotation_deg": float(
                    np.nanmean(
                        [
                            row["aggregate"].get(
                                "wrist_delta_rotation_error_deg", math.nan
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_jitter_m": float(
                    np.nanmean(
                        [
                            row["aggregate"].get("predicted_action_jitter_m", math.nan)
                            for row in selected
                        ]
                    )
                ),
                "mean_coverage": float(
                    np.nanmean(
                        [row["aggregate"].get("coverage", math.nan) for row in selected]
                    )
                ),
            }
    (RUN / "scores.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (RUN / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary


def add_world_scores(
    rows: list[dict[str, Any]],
    clips: list[str],
    predictions: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    experiment: str,
) -> None:
    for clip in clips:
        prediction = predictions.get(clip)
        if prediction is None:
            continue
        for horizon in HORIZONS:
            score = base.score(
                prediction, targets[clip], horizon=horizon, key="T_world_wrist"
            )
            rows.append(
                {
                    "clip": clip,
                    "experiment": experiment,
                    "horizon": horizon,
                    "score": score.get("score"),
                    "aggregate": score.get("aggregate"),
                    "hands": score.get("hands"),
                }
            )


def ensure_vggt_world(prediction: dict[str, Any], camera: np.ndarray) -> dict[str, Any]:
    return vggt_world_mod.project_with_camera(prediction, camera)


def run_temporal_loo(
    rows: list[dict[str, Any]],
    clips: list[str],
    predictions: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    experiment: str,
) -> None:
    model_predictions = {
        clip: vggt_world_mod.world_model_payload(pred)
        for clip, pred in predictions.items()
    }
    model_targets = {
        clip: vggt_world_mod.world_model_payload(target)
        for clip, target in targets.items()
    }
    for holdout in clips:
        if holdout not in model_predictions:
            continue
        models = {}
        for side in HANDS:
            x, y = state.collect_train(
                model_predictions, model_targets, holdout, side, "temporal", "direct"
            )
            models[side] = state.fit_standardized_ridge(x, y, 0.01)
        corrected_model = state.apply_model(
            model_predictions[holdout],
            models,
            feature_mode="temporal",
            target_mode="direct",
            blend=0.5,
        )
        corrected = vggt_world_mod.from_world_model_payload(corrected_model)
        for horizon in HORIZONS:
            score = base.score(
                corrected, targets[holdout], horizon=horizon, key="T_world_wrist"
            )
            rows.append(
                {
                    "clip": holdout,
                    "experiment": experiment,
                    "horizon": horizon,
                    "score": score.get("score"),
                    "aggregate": score.get("aggregate"),
                    "hands": score.get("hands"),
                }
            )


def temporal_experiment(
    rows: list[dict[str, Any]],
    clips: list[str],
    predictions: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    experiment: str,
) -> None:
    add_world_scores(rows, clips, predictions, targets, f"{experiment}_raw")
    run_temporal_loo(rows, clips, predictions, targets, f"{experiment}_temporal")


def candidate_from_op(
    clips: list[str],
    base_predictions: dict[str, dict[str, Any]],
    cameras: dict[str, np.ndarray],
    op: Callable[[str, dict[str, Any]], dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for clip in clips:
        prediction = base_predictions.get(clip)
        if prediction is None:
            continue
        corrected = op(clip, prediction)
        if corrected is None:
            continue
        out[clip] = ensure_vggt_world(corrected, cameras[clip])
    return out


def combine_ops(
    clips: list[str],
    base_predictions: dict[str, dict[str, Any]],
    cameras: dict[str, np.ndarray],
    ops: list[Callable[[str, dict[str, Any]], dict[str, Any] | None]],
) -> dict[str, dict[str, Any]]:
    def apply_all(clip: str, prediction: dict[str, Any]) -> dict[str, Any] | None:
        current: dict[str, Any] | None = prediction
        for op in ops:
            if current is None:
                return None
            current = op(clip, current)
        return current

    return candidate_from_op(clips, base_predictions, cameras, apply_all)


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    clips = sorted(path.name for path in SRC.glob("clip-*") if path.is_dir())[:10]
    raw = {
        clip: json.loads((SRC / clip / "hawor_result.json").read_text(encoding="utf-8"))
        for clip in clips
    }
    targets = {clip: base.gt(base.RAW / f"{clip}.tar") for clip in clips}
    cameras = {clip: vggt_world_mod.load_camera(clip) for clip in clips}
    rows: list[dict[str, Any]] = []
    log("vggt_world_aoe_fixes_start", clips=clips)

    prepared_tw7 = {
        clip: state.prepare(payload, 7, 0.675, 9, 0.8) for clip, payload in raw.items()
    }
    baseline_world = {
        clip: ensure_vggt_world(prepared_tw7[clip], cameras[clip]) for clip in clips
    }
    temporal_experiment(rows, clips, baseline_world, targets, "baseline_vggt_world_tw7")
    summarize(rows, clips)
    log("baseline_done")

    # 1. Depth-rescaled HaWoR translation, using VGGT depth and optional MoGe depth when cached.
    depth_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for backend in ("vggt", "moge"):
        for mode in ("wrist", "palm", "all"):
            for blend in (0.10, 0.25, 0.50):
                for clamp in ((0.9, 1.1), (0.75, 1.25)):
                    experiment = (
                        f"depth_{backend}_{mode}_b{blend:g}_c{clamp[0]:g}_{clamp[1]:g}"
                    )
                    predictions = candidate_from_op(
                        clips,
                        prepared_tw7,
                        cameras,
                        lambda clip, pred, backend=backend, mode=mode, blend=blend, clamp=clamp: (
                            remaining.depth_rescale(
                                pred,
                                clip,
                                backend=backend,
                                mode=mode,
                                blend=blend,
                                clamp=clamp,
                            )
                        ),
                    )
                    if not predictions:
                        continue
                    depth_candidates[experiment] = predictions
                    temporal_experiment(rows, clips, predictions, targets, experiment)
        summarize(rows, clips)
        log("depth_backend_done", backend=backend)

    # 2. Joint velocity, Z jump, hand-scale, and reprojection outlier filtering.
    filter_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for sigma in (2.0, 2.5, 3.0, 3.5):
        for repair in ("interp", "mask"):
            for name, fn in (
                ("velocity", aoe.velocity_filter),
                ("zjump", remaining.z_jump_filter),
                ("handscale", remaining.hand_scale_filter),
            ):
                experiment = f"filter_{name}_s{sigma:g}_{repair}"
                predictions = candidate_from_op(
                    clips,
                    prepared_tw7,
                    cameras,
                    lambda _clip, pred, fn=fn, sigma=sigma, repair=repair: fn(
                        pred, sigma=sigma, repair=repair
                    ),
                )
                filter_candidates[experiment] = predictions
                temporal_experiment(rows, clips, predictions, targets, experiment)
    for threshold_px in (40.0, 80.0, 120.0):
        for repair in ("interp", "mask"):
            experiment = f"filter_reprojection_thr{threshold_px:g}_{repair}"
            predictions = candidate_from_op(
                clips,
                prepared_tw7,
                cameras,
                lambda clip, pred, threshold_px=threshold_px, repair=repair: (
                    aoe.bbox_residual_filter(
                        pred,
                        clip,
                        threshold_px=threshold_px,
                        repair=repair,
                    )
                ),
            )
            filter_candidates[experiment] = predictions
            temporal_experiment(rows, clips, predictions, targets, experiment)
    summarize(rows, clips)
    log("filters_done")

    # 3. Acceleration-regularized window optimization with stronger kinematic smoothness.
    window_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for accel_weight in (0.5, 1.0, 2.5, 5.0, 10.0, 20.0):
        for jerk_weight in (0.0, 0.05, 0.1, 0.5):
            for alpha in (0.625, 0.675, 0.7):
                experiment = f"window_accel{accel_weight:g}_jerk{jerk_weight:g}_a{alpha:g}_rw9_g0.8"
                predictions = candidate_from_op(
                    clips,
                    raw,
                    cameras,
                    lambda _clip, pred, accel_weight=accel_weight, jerk_weight=jerk_weight, alpha=alpha: (
                        remaining.window_optimize(
                            pred,
                            alpha=alpha,
                            rw=9,
                            gamma=0.8,
                            accel_weight=accel_weight,
                            jerk_weight=jerk_weight,
                        )
                    ),
                )
                window_candidates[experiment] = predictions
                temporal_experiment(rows, clips, predictions, targets, experiment)
        summarize(rows, clips)
        log("window_accel_done", accel_weight=accel_weight)

    # 4. Combined candidates around the best prior non-world AoE settings.
    combined_specs = {
        "combo_window5_depth_vggt_all010_filter_vel35": [
            lambda _clip, pred: remaining.window_optimize(
                pred, alpha=0.625, rw=9, gamma=0.8, accel_weight=5.0, jerk_weight=0.0
            ),
            lambda clip, pred: remaining.depth_rescale(
                pred, clip, backend="vggt", mode="all", blend=0.10, clamp=(0.9, 1.1)
            ),
            lambda _clip, pred: aoe.velocity_filter(pred, sigma=3.5, repair="interp"),
        ],
        "combo_window10_filter_vel35": [
            lambda _clip, pred: remaining.window_optimize(
                pred, alpha=0.625, rw=9, gamma=0.8, accel_weight=10.0, jerk_weight=0.0
            ),
            lambda _clip, pred: aoe.velocity_filter(pred, sigma=3.5, repair="interp"),
        ],
        "combo_filter_vel35_depth_vggt_all010": [
            lambda _clip, pred: aoe.velocity_filter(pred, sigma=3.5, repair="interp"),
            lambda clip, pred: remaining.depth_rescale(
                pred, clip, backend="vggt", mode="all", blend=0.10, clamp=(0.9, 1.1)
            ),
        ],
    }
    for experiment, ops in combined_specs.items():
        predictions = combine_ops(
            clips,
            raw if experiment.startswith("combo_window") else prepared_tw7,
            cameras,
            ops,
        )
        if not predictions:
            continue
        temporal_experiment(rows, clips, predictions, targets, experiment)
    summary = summarize(rows, clips)
    log("combined_done")

    best = []
    for experiment, horizons in summary["experiments"].items():
        h1 = horizons.get("1")
        if h1:
            best.append((h1["mean_score"], experiment, h1))
    for score, experiment, h1 in sorted(best)[:80]:
        print(
            f"BEST {score:.3f} {experiment} n={h1['count']} "
            f"trans={h1['mean_translation_m']:.5f} rot={h1['mean_rotation_deg']:.3f} "
            f"jit={h1['mean_jitter_m']:.5f} cov={h1['mean_coverage']:.3f}",
            flush=True,
        )
    log("vggt_world_aoe_fixes_done", summary=str(RUN / "summary.json"))


if __name__ == "__main__":
    main()
