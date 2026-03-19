from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pyarrow as pa

StatValue: TypeAlias = list["StatValue"] | bool | int | float
_DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


@dataclass(frozen=True, slots=True)
class LeRobotFeatureStats:
    min: StatValue | None = None
    max: StatValue | None = None
    mean: StatValue | None = None
    std: StatValue | None = None
    count: StatValue | None = None
    q01: StatValue | None = None
    q10: StatValue | None = None
    q50: StatValue | None = None
    q90: StatValue | None = None
    q99: StatValue | None = None

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, StatValue | None],
    ) -> "LeRobotFeatureStats":
        return cls(
            min=payload.get("min"),
            max=payload.get("max"),
            mean=payload.get("mean"),
            std=payload.get("std"),
            count=payload.get("count"),
            q01=payload.get("q01"),
            q10=payload.get("q10"),
            q50=payload.get("q50"),
            q90=payload.get("q90"),
            q99=payload.get("q99"),
        )

    def to_json_dict(self) -> dict[str, StatValue | None]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "q01": self.q01,
            "q10": self.q10,
            "q50": self.q50,
            "q90": self.q90,
            "q99": self.q99,
        }


@dataclass(frozen=True, slots=True)
class LeRobotStatsFile(Mapping[str, LeRobotFeatureStats]):
    features: Mapping[str, LeRobotFeatureStats]

    def __getitem__(self, key: str) -> LeRobotFeatureStats:
        return self.features[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def flatten_fields(self) -> dict[str, StatValue | None]:
        return {
            f"stats/{feature}/{field}": value
            for feature, feature_stats in self.items()
            for field, value in feature_stats.to_json_dict().items()
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Mapping[str, StatValue | None]],
    ) -> "LeRobotStatsFile":
        return cls(
            {
                feature: LeRobotFeatureStats.from_json_dict(stats)
                for feature, stats in payload.items()
                if isinstance(feature, str) and isinstance(stats, Mapping)
            }
        )

    def to_json_dict(self) -> dict[str, dict[str, StatValue | None]]:
        return {
            feature: feature_stats.to_json_dict()
            for feature, feature_stats in self.items()
        }


def compute_feature_stats(array: np.ndarray) -> LeRobotFeatureStats:
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)

    count = int(array.shape[0])
    if count < 2:
        mean = np.atleast_1d(np.mean(array, axis=0)).tolist()
        return LeRobotFeatureStats(
            min=np.atleast_1d(np.min(array, axis=0)).tolist(),
            max=np.atleast_1d(np.max(array, axis=0)).tolist(),
            mean=mean,
            std=np.atleast_1d(np.std(array, axis=0)).tolist(),
            count=np.array([count], dtype=np.int64).tolist(),
            q01=list(mean),
            q10=list(mean),
            q50=list(mean),
            q90=list(mean),
            q99=list(mean),
        )

    running = _RunningQuantileStats(
        quantile_list=list(_DEFAULT_QUANTILES),
        num_quantile_bins=5000,
    )
    running.update(array)
    stats = running.get_statistics()
    return LeRobotFeatureStats(
        min=stats["min"].tolist(),
        max=stats["max"].tolist(),
        mean=stats["mean"].tolist(),
        std=stats["std"].tolist(),
        count=stats["count"].tolist(),
        q01=stats["q01"].tolist(),
        q10=stats["q10"].tolist(),
        q50=stats["q50"].tolist(),
        q90=stats["q90"].tolist(),
        q99=stats["q99"].tolist(),
    )


def compute_table_stats(table: pa.Table) -> LeRobotStatsFile:
    out: dict[str, LeRobotFeatureStats] = {}
    for key in table.column_names:
        # Skip routing/bookkeeping columns that are labels or synthetic indices,
        # not real measured features we want dataset stats for.
        if key in {"index", "episode_index", "task_index", "frame_index", "timestamp"}:
            continue

        column = table.column(key)
        numeric = _numeric_column(column)
        if numeric is None:
            numeric = _stack_numeric_values(column.to_pylist())
        if numeric is None:
            continue
        out[key] = compute_feature_stats(numeric)
    return LeRobotStatsFile(out)


def _numeric_column(column: pa.ChunkedArray) -> np.ndarray | None:
    if column.null_count > 0:
        return None

    if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
        values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values if values.size and np.isfinite(values).all() else None

    try:
        combined = column.combine_chunks()
    except (TypeError, ValueError):
        return None

    if pa.types.is_fixed_size_list(combined.type):
        width = int(combined.type.list_size)
    elif pa.types.is_list(combined.type) or pa.types.is_large_list(combined.type):
        offsets = np.asarray(combined.offsets.to_numpy(zero_copy_only=False))
        lengths = np.diff(offsets)
        if lengths.size == 0 or np.any(lengths != lengths[0]) or lengths[0] <= 0:
            return None
        width = int(lengths[0])
    else:
        return None

    value_type = combined.type.value_type
    if not (pa.types.is_integer(value_type) or pa.types.is_floating(value_type)):
        return None
    if combined.values.null_count > 0:
        return None

    values = np.asarray(
        combined.values.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if values.size == 0 or not np.isfinite(values).all():
        return None
    row_count = len(combined)
    return (
        values.reshape(row_count, width) if values.size == row_count * width else None
    )


def _stack_numeric_values(values: list[object]) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for value in values:
        if value is None or isinstance(value, bool):
            return None
        array = np.asarray(value)
        if array.size == 0 or array.dtype.kind not in {"i", "u", "f"}:
            return None
        numeric = array.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            return None
        arrays.append(numeric)

    if not arrays:
        return None
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        return None
    return np.stack(arrays, axis=0)


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


__all__ = [
    "compute_table_stats",
    "LeRobotFeatureStats",
    "LeRobotStatsFile",
    "StatValue",
    "compute_feature_stats",
]
