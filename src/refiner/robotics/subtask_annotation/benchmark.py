from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from refiner.robotics.subtask_annotation.evaluation import (
    SubtaskSegmentationMetrics,
    evaluate_subtask_segments,
)


_HIGHER_IS_BETTER = {
    "r_at_50",
    "r_at_70",
    "precision",
    "recall",
    "f1",
    "mean_iou",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
}
_LOWER_IS_BETTER = {
    "boundary_mae_s",
    "overseg_error",
    "creates",
    "deletes",
    "drags",
    "edit_cost_per_min",
}


def run_segmentation_benchmark(
    *,
    gold_path: str | Path,
    candidate_paths: Mapping[str, str | Path],
    baseline: str | None = None,
    output_path: str | Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate candidate JSON files against one frozen gold-set JSON file."""

    if not candidate_paths:
        raise ValueError("candidate_paths must not be empty")
    if baseline is not None and baseline not in candidate_paths:
        raise ValueError(f"baseline {baseline!r} is not a candidate")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")

    gold_bytes = Path(gold_path).read_bytes()
    gold = _records(json.loads(gold_bytes), source="gold")
    candidates: dict[str, dict[str, Mapping[str, Any]]] = {}
    digests = {"gold": _sha256(gold_bytes), "candidates": {}}
    for name, path in candidate_paths.items():
        if not name.strip():
            raise ValueError("candidate names must be non-empty")
        raw = Path(path).read_bytes()
        candidates[name] = _records(json.loads(raw), source=f"candidate {name!r}")
        digests["candidates"][name] = _sha256(raw)

    report = benchmark_segmentation(
        gold,
        candidates,
        baseline=baseline,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    report["input_sha256"] = digests
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def benchmark_segmentation(
    gold: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    baseline: str | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate normalized record mappings and return a JSON-safe report."""

    if not gold:
        raise ValueError("gold set must not be empty")
    if not candidates:
        raise ValueError("candidates must not be empty")
    if baseline is not None and baseline not in candidates:
        raise ValueError(f"baseline {baseline!r} is not a candidate")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")

    per_candidate: dict[str, dict[str, Any]] = {}
    metric_rows: dict[str, dict[str, dict[str, float | None]]] = {}
    for candidate_name, candidate_records in candidates.items():
        rows = []
        metric_rows[candidate_name] = {}
        missing_ids = []
        invalid_ids = []
        blocked_ids = []
        latencies = []
        token_counts = []
        for video_id, gold_record in gold.items():
            duration_s = _duration(gold_record, video_id=video_id)
            reference = _segments(gold_record, video_id=video_id, gold=True)
            candidate = candidate_records.get(video_id)
            if candidate is None:
                missing_ids.append(video_id)
                predicted: Sequence[Mapping[str, Any]] = []
                status = "missing"
            else:
                status = _status(candidate)
                if status == "blocked":
                    blocked_ids.append(video_id)
                if status in {"blocked", "invalid"}:
                    predicted = []
                    if status == "invalid":
                        invalid_ids.append(video_id)
                else:
                    try:
                        predicted = _segments(candidate, video_id=video_id, gold=False)
                    except ValueError:
                        predicted = []
                        status = "invalid"
                        invalid_ids.append(video_id)
                latency = _latency_ms(candidate)
                if latency is not None:
                    latencies.append(latency)
                tokens = _tokens(candidate)
                if tokens is not None:
                    token_counts.append(tokens)

            metrics = evaluate_subtask_segments(
                predicted,
                reference,
                video_duration_s=duration_s,
            )
            metrics_dict = metrics.model_dump(mode="json")
            metrics_dict["overseg_error"] = (
                abs(float(metrics.overseg_ratio) - 1.0)
                if metrics.overseg_ratio is not None
                else None
            )
            metric_rows[candidate_name][video_id] = metrics_dict
            rows.append(
                {
                    "video_id": video_id,
                    "status": status,
                    "duration_s": duration_s,
                    "gold_segments": len(reference),
                    "predicted_segments": len(predicted),
                    "metrics": metrics_dict,
                }
            )

        unexpected = sorted(set(candidate_records).difference(gold))
        per_candidate[candidate_name] = {
            "aggregate": _aggregate_metrics(list(metric_rows[candidate_name].values())),
            "coverage": {
                "gold_videos": len(gold),
                "predicted_videos": len(gold) - len(missing_ids),
                "missing_rate": len(missing_ids) / len(gold),
                "invalid_rate": len(invalid_ids) / len(gold),
                "blocked_rate": len(blocked_ids) / len(gold),
                "missing_video_ids": missing_ids,
                "invalid_video_ids": invalid_ids,
                "blocked_video_ids": blocked_ids,
                "unexpected_video_ids": unexpected,
            },
            "runtime": {
                "mean_latency_ms": _mean(latencies),
                "mean_tokens": _mean(token_counts),
            },
            "videos": rows,
        }

    comparisons = {}
    if baseline is not None:
        for candidate_name in candidates:
            if candidate_name == baseline:
                continue
            comparisons[candidate_name] = _paired_comparison(
                baseline=metric_rows[baseline],
                candidate=metric_rows[candidate_name],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )

    return {
        "schema_version": 1,
        "baseline": baseline,
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "candidates": per_candidate,
        "paired_vs_baseline": comparisons,
    }


def _records(value: Any, *, source: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping) and "records" in value:
        value = value["records"]
    if isinstance(value, Mapping):
        items = []
        for key, record in value.items():
            if not isinstance(record, Mapping):
                raise ValueError(f"{source} record {key!r} must be an object")
            merged = dict(record)
            merged.setdefault("video_id", str(key))
            items.append(merged)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        raise ValueError(f"{source} must be a record list, mapping, or records object")

    records: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(items):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}[{index}] must be an object")
        raw_id = record.get(
            "video_id", record.get("episode_id", record.get("data_hash"))
        )
        video_id = str(raw_id or "").strip()
        if not video_id:
            raise ValueError(f"{source}[{index}] is missing video_id")
        if video_id in records:
            raise ValueError(f"{source} contains duplicate video_id {video_id!r}")
        records[video_id] = dict(cast(Mapping[str, Any], record))
    return records


def _duration(record: Mapping[str, Any], *, video_id: str) -> float:
    try:
        duration = float(record["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"gold video {video_id!r} requires numeric duration_s"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"gold video {video_id!r} duration_s must be finite and > 0")
    return duration


def _segments(
    record: Mapping[str, Any],
    *,
    video_id: str,
    gold: bool,
) -> Sequence[Mapping[str, Any]]:
    keys = (
        ("segments", "reference_segments", "gold_segments")
        if gold
        else ("segments", "predicted_subtasks")
    )
    value: Any = None
    found = False
    for key in keys:
        if key in record:
            value = record[key]
            found = True
            break
    if not found and not gold:
        result = record.get("subtask_annotation_result")
        if isinstance(result, Mapping) and "segments" in result:
            value = result["segments"]
            found = True
    if not found or value is None:
        if not gold and _status(record) in {"blocked", "invalid"}:
            return []
        raise ValueError(f"video {video_id!r} is missing segments")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"video {video_id!r} segments must be a list")
    if any(not isinstance(segment, Mapping) for segment in value):
        raise ValueError(f"video {video_id!r} segments must contain objects")
    return value


def _result(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("subtask_annotation_result")
    return value if isinstance(value, Mapping) else record


def _status(record: Mapping[str, Any]) -> str:
    return str(_result(record).get("status") or "ok")


def _latency_ms(record: Mapping[str, Any]) -> float | None:
    provenance = _result(record).get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    return _finite_optional(provenance.get("latency_ms"))


def _tokens(record: Mapping[str, Any]) -> float | None:
    provenance = _result(record).get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    usage = provenance.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for key in ("totalTokens", "total_tokens", "totalTokenCount"):
        value = _finite_optional(usage.get(key))
        if value is not None:
            return value
    values = [
        _finite_optional(usage.get(key))
        for key in (
            "inputTokens",
            "outputTokens",
            "input_tokens",
            "output_tokens",
            "promptTokenCount",
            "candidatesTokenCount",
        )
    ]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _aggregate_metrics(rows: Sequence[Mapping[str, float | None]]) -> dict[str, Any]:
    keys = tuple(SubtaskSegmentationMetrics.model_fields) + ("overseg_error",)
    return {
        key: _mean(
            [_metric_value(row, key) for row in rows if row.get(key) is not None]
        )
        for key in keys
    }


def _paired_comparison(
    *,
    baseline: Mapping[str, Mapping[str, float | None]],
    candidate: Mapping[str, Mapping[str, float | None]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    metric_names = sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER)
    for metric_index, metric in enumerate(metric_names):
        paired = [
            (
                _metric_value(baseline[video_id], metric),
                _metric_value(candidate[video_id], metric),
            )
            for video_id in baseline
            if baseline[video_id].get(metric) is not None
            and candidate[video_id].get(metric) is not None
        ]
        if not paired:
            output[metric] = None
            continue
        raw_deltas = [right - left for left, right in paired]
        improvements = (
            raw_deltas
            if metric in _HIGHER_IS_BETTER
            else [-delta for delta in raw_deltas]
        )
        samples = _bootstrap_means(
            improvements,
            samples=bootstrap_samples,
            seed=seed + metric_index,
        )
        output[metric] = {
            "n": len(paired),
            "candidate_minus_baseline": _mean(raw_deltas),
            "mean_improvement": _mean(improvements),
            "improvement_ci_95": [
                _quantile(samples, 0.025),
                _quantile(samples, 0.975),
            ],
            "probability_improved": sum(value > 0 for value in samples) / len(samples),
        }
    return output


def _bootstrap_means(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    return [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _finite_optional(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(row: Mapping[str, float | None], key: str) -> float:
    value = row[key]
    if value is None:
        raise ValueError(f"metric {key!r} is missing")
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark subtask segmentation candidates against frozen gold JSON."
    )
    parser.add_argument("--gold", required=True, help="Gold-set JSON path")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Candidate JSON; repeat for multiple systems",
    )
    parser.add_argument("--baseline", help="Candidate name used for paired deltas")
    parser.add_argument("--out", required=True, help="Report JSON path")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    candidates = {}
    for value in args.candidate:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("--candidate must use NAME=PATH")
        if name in candidates:
            parser.error(f"duplicate candidate name: {name}")
        candidates[name] = path
    if not candidates:
        parser.error("at least one --candidate is required")

    report = run_segmentation_benchmark(
        gold_path=args.gold,
        candidate_paths=candidates,
        baseline=args.baseline,
        output_path=args.out,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {name: data["aggregate"] for name, data in report["candidates"].items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["benchmark_segmentation", "main", "run_segmentation_benchmark"]
