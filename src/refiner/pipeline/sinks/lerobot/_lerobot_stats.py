from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


@dataclass(slots=True)
class _RunningQuantileStats:
    quantile_list: list[float]
    num_quantile_bins: int = 5000
    _count: int = 0
    _mean: np.ndarray | None = None
    _mean_of_squares: np.ndarray | None = None
    _min: np.ndarray | None = None
    _max: np.ndarray | None = None
    _histograms: list[np.ndarray] | None = None
    _bin_edges: list[np.ndarray] | None = None

    @property
    def count(self) -> int:
        return int(self._count)

    def update(self, batch: np.ndarray) -> None:
        if batch.ndim != 2:
            raise ValueError("batch must be 2D (N, C)")
        if batch.shape[0] == 0:
            return

        batch = batch.astype(np.float64, copy=False)
        batch_rows, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self.num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self.num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if (
                self._mean is None
                or self._mean_of_squares is None
                or self._min is None
                or self._max is None
            ):
                raise RuntimeError("RunningQuantileStats state is not initialized")
            if vector_length != self._mean.size:
                raise ValueError("batch channel dimension mismatch")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            needs_rebucket = np.any(new_max > self._max) or np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if needs_rebucket:
                self._adjust_histograms()

        self._count += batch_rows
        if self._mean is None or self._mean_of_squares is None:
            raise RuntimeError("RunningQuantileStats state is not initialized")

        weight = batch_rows / self._count
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * weight
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            weight
        )
        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        if (
            self._count <= 0
            or self._mean is None
            or self._min is None
            or self._max is None
        ):
            raise ValueError("Cannot compute stats without samples")

        variance = (
            self._mean_of_squares - self._mean**2
            if self._mean_of_squares is not None
            else np.zeros_like(self._mean)
        )
        stats: dict[str, np.ndarray] = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": np.sqrt(np.maximum(0.0, variance)),
            "count": np.array([self._count], dtype=np.int64),
        }
        if self._count < 2:
            for q in self.quantile_list:
                stats[f"q{int(q * 100):02d}"] = self._mean.copy()
            return stats

        quantiles = self._compute_quantiles()
        for i, q in enumerate(self.quantile_list):
            stats[f"q{int(q * 100):02d}"] = quantiles[i]
        return stats

    def _adjust_histograms(self) -> None:
        if (
            self._histograms is None
            or self._bin_edges is None
            or self._min is None
            or self._max is None
        ):
            return

        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding,
                self._max[i] + padding,
                self.num_quantile_bins + 1,
            )
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self.num_quantile_bins)
            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count <= 0:
                    continue
                bin_idx = np.searchsorted(new_edges, old_center) - 1
                bin_idx = max(0, min(bin_idx, self.num_quantile_bins - 1))
                new_hist[bin_idx] += count
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        if self._histograms is None or self._bin_edges is None:
            return
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        if self._histograms is None or self._bin_edges is None:
            raise RuntimeError("RunningQuantileStats histograms are not initialized")

        out: list[np.ndarray] = []
        for q in self.quantile_list:
            target_count = q * self._count
            out.append(
                np.asarray(
                    [
                        self._compute_single_quantile(hist, edges, target_count)
                        for hist, edges in zip(
                            self._histograms, self._bin_edges, strict=True
                        )
                    ]
                )
            )
        return out

    @staticmethod
    def _compute_single_quantile(
        hist: np.ndarray,
        edges: np.ndarray,
        target_count: float,
    ) -> float:
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)
        if idx == 0:
            return float(edges[0])
        if idx >= len(cumsum):
            return float(edges[-1])

        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before
        if count_in_bin == 0:
            return float(edges[idx])

        fraction = (target_count - count_before) / count_in_bin
        return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def _feature_stats(
    array: np.ndarray,
    *,
    num_quantile_bins: int = 5000,
) -> dict[str, np.ndarray]:
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)

    count = int(array.shape[0])
    if count < 2:
        mean = np.atleast_1d(np.mean(array, axis=0))
        stats = {
            "min": np.atleast_1d(np.min(array, axis=0)),
            "max": np.atleast_1d(np.max(array, axis=0)),
            "mean": mean,
            "std": np.atleast_1d(np.std(array, axis=0)),
            "count": np.array([count], dtype=np.int64),
        }
        for q in _DEFAULT_QUANTILES:
            stats[f"q{int(q * 100):02d}"] = mean.copy()
        return stats

    running_stats = _RunningQuantileStats(
        quantile_list=list(_DEFAULT_QUANTILES),
        num_quantile_bins=num_quantile_bins,
    )
    running_stats.update(array)
    return running_stats.get_statistics()


def _flatten_stats_for_episode(
    stats: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature, feature_stats in stats.items():
        for stat_name, stat_value in feature_stats.items():
            out[f"stats/{feature}/{stat_name}"] = _jsonable_value(stat_value)
    return out


def _extract_episode_stats(
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        feature: feature_stats
        for feature, feature_stats in _extract_episode_stats_raw(row).items()
        if feature_stats is not None
    }


def _extract_episode_stats_raw(
    row: Mapping[str, Any],
    *,
    preserve_null: bool = False,
) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    legacy_stats: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            continue
        if _is_grouped_episode_stats_column(key):
            if value is None:
                if preserve_null:
                    out[_feature_name_from_stats_column(key)] = None
                continue
            if not isinstance(value, Mapping):
                continue
            out[_feature_name_from_stats_column(key)] = {
                str(stat_name): stat_value for stat_name, stat_value in value.items()
            }
            continue
        if not key.startswith("stats/"):
            continue
        parts = key.split("/", 2)
        if len(parts) != 3:
            continue
        _, feature, stat_name = parts
        legacy_stats.setdefault(feature, {})[stat_name] = value
    for feature, feature_stats in legacy_stats.items():
        out.setdefault(feature, feature_stats)
    return out


def _aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    if not stats_list:
        return {}

    feature_keys = {k for stats in stats_list for k in stats.keys()}
    out: dict[str, dict[str, np.ndarray]] = {}

    for feature in sorted(feature_keys):
        items = [stats[feature] for stats in stats_list if feature in stats]
        means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in items])
        stds = np.stack([np.asarray(item["std"], dtype=np.float64) for item in items])
        counts = np.stack(
            [np.asarray(item["count"], dtype=np.float64) for item in items]
        )
        mins = np.stack([np.asarray(item["min"], dtype=np.float64) for item in items])
        maxs = np.stack([np.asarray(item["max"], dtype=np.float64) for item in items])

        total_count = counts.sum(axis=0)
        weights = counts.reshape(counts.shape + (1,) * (means.ndim - counts.ndim))

        total_mean = (means * weights).sum(axis=0) / np.maximum(total_count, 1e-12)
        variances = stds**2
        delta = means - total_mean
        total_var = ((variances + delta**2) * weights).sum(axis=0) / np.maximum(
            total_count,
            1e-12,
        )

        agg: dict[str, np.ndarray] = {
            "min": np.min(mins, axis=0),
            "max": np.max(maxs, axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_var),
            "count": total_count,
        }

        quantile_keys = [
            k for k in items[0].keys() if k.startswith("q") and k[1:].isdigit()
        ]
        for q_key in quantile_keys:
            if not all(q_key in item for item in items):
                continue
            q_vals = np.stack(
                [np.asarray(item[q_key], dtype=np.float64) for item in items]
            )
            agg[q_key] = (q_vals * weights).sum(axis=0) / np.maximum(
                total_count,
                1e-12,
            )

        out[feature] = agg

    return out


def _serialize_stats(
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for feature, feature_stats in stats.items():
        out_feature: dict[str, Any] = {}
        for key, value in feature_stats.items():
            out_feature[key] = _jsonable_value(value)
        out[feature] = out_feature
    return out


def _cast_stats_to_numpy(
    raw: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray] | None]:
    out: dict[str, dict[str, np.ndarray] | None] = {}
    for feature, feature_stats in raw.items():
        if feature_stats is None:
            out[str(feature)] = None
            continue
        if not isinstance(feature_stats, Mapping):
            continue
        inner: dict[str, np.ndarray] = {}
        for key, value in feature_stats.items():
            inner[str(key)] = np.asarray(value)
        out[str(feature)] = inner
    return out


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _episode_stats_column_name(feature: str) -> str:
    return f"stats/{feature}"


def _feature_name_from_stats_column(column_name: str) -> str:
    _, feature = column_name.split("/", 1)
    return feature


def _is_grouped_episode_stats_column(column_name: str) -> bool:
    return column_name.startswith("stats/") and column_name.count("/") == 1
