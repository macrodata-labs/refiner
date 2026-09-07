from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import math
import sys
import time
from typing import Any, Generic, TextIO, TypeVar

from refiner.worker.context import (
    get_active_step_index,
    get_active_user_metrics_emitter,
)


T = TypeVar("T")


def _length(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _metric_name(description: str | None) -> str:
    normalized = " ".join((description or "").split())
    return f"progress.{normalized}" if normalized else "progress"


def _format_interval(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


class Progress(Generic[T]):
    """Track iterable or manually updated work with tqdm-style semantics.

    Progress is emitted as absolute gauges, so a retried worker attempt does not
    replay previously emitted increments. Interactive local runs also render a
    compact terminal progress line.
    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        *,
        total: float | None = None,
        desc: str | None = None,
        unit: str = "it",
        initial: float = 0,
        mininterval: float = 0.5,
        file: TextIO | None = None,
        disable: bool | None = None,
    ) -> None:
        if total is None and iterable is not None:
            total = _length(iterable)
        if total is not None and total < 0:
            raise ValueError("total must be >= 0")
        if initial < 0:
            raise ValueError("initial must be >= 0")
        if mininterval < 0:
            raise ValueError("mininterval must be >= 0")
        if not unit.strip():
            raise ValueError("unit must be non-empty")

        self.iterable = iterable
        self.total = float(total) if total is not None else None
        self.desc = desc or ""
        self.unit = unit
        self.n = float(initial)
        self.mininterval = float(mininterval)
        self.file = file or sys.stderr
        self.disable = (
            not bool(getattr(self.file, "isatty", lambda: False)())
            if disable is None
            else disable
        )
        self.postfix: dict[str, Any] = {}
        self._metric_label = _metric_name(desc)
        self._started_at = time.monotonic()
        self._last_refresh_at: float | None = None
        self._last_render_width = 0
        self._closed = False
        self.refresh()

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def rate(self) -> float | None:
        elapsed = self.elapsed
        if elapsed <= 0:
            return None
        return max(0.0, self.n / elapsed)

    def update(self, n: float = 1) -> None:
        if self._closed:
            return
        self.n += n
        self.refresh()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        self.desc = desc or ""
        if refresh:
            self.refresh(force=True)

    def set_postfix(
        self,
        ordered_dict: Mapping[str, Any] | None = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        self.postfix = dict(ordered_dict or {})
        self.postfix.update(kwargs)
        if refresh:
            self.refresh(force=True)

    def _emit_metrics(self, *, elapsed: float, rate: float | None) -> None:
        emitter = get_active_user_metrics_emitter()
        step_index = get_active_step_index()
        values: dict[str, float] = {
            "completed": self.n,
            "elapsed_seconds": elapsed,
        }
        if self.total is not None:
            values["total"] = self.total
        if rate is not None:
            values["rate"] = rate
        for key, value in self.postfix.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                values[f"postfix.{key}"] = float(value)
        for kind, value in values.items():
            emitter.emit_user_gauge(
                label=self._metric_label,
                value=float(value),
                kind=kind,
                step_index=step_index,
                unit=(
                    f"{self.unit}/s"
                    if kind == "rate"
                    else self.unit
                    if kind in {"completed", "total"}
                    else None
                ),
            )

    def _render(self, *, elapsed: float, rate: float | None) -> None:
        if self.disable:
            return
        prefix = f"{self.desc}: " if self.desc else ""
        count = f"{self.n:g}"
        if self.total is None:
            amount = count
            percent = ""
            remaining = None
        else:
            amount = f"{count}/{self.total:g}"
            percent = f" {0 if self.total == 0 else self.n / self.total:6.1%}"
            remaining = (
                None
                if rate is None or rate <= 0
                else max(0.0, self.total - self.n) / rate
            )
        rate_text = "?" if rate is None else f"{rate:.2f}"
        postfix = ", ".join(f"{key}={value}" for key, value in self.postfix.items())
        text = (
            f"{prefix}{amount} {self.unit}{percent} "
            f"[{_format_interval(elapsed)}<{_format_interval(remaining)}, "
            f"{rate_text} {self.unit}/s]"
        )
        if postfix:
            text += f" {postfix}"
        padding = " " * max(0, self._last_render_width - len(text))
        self.file.write(f"\r{text}{padding}")
        self.file.flush()
        self._last_render_width = len(text)

    def refresh(self, *, force: bool = False) -> bool:
        if self._closed:
            return False
        now = time.monotonic()
        if (
            not force
            and self._last_refresh_at is not None
            and now - self._last_refresh_at < self.mininterval
        ):
            return False
        elapsed = max(0.0, now - self._started_at)
        rate = None if elapsed <= 0 else max(0.0, self.n / elapsed)
        self._emit_metrics(elapsed=elapsed, rate=rate)
        self._render(elapsed=elapsed, rate=rate)
        self._last_refresh_at = now
        return True

    def close(self) -> None:
        if self._closed:
            return
        self.refresh(force=True)
        self._closed = True
        if not self.disable:
            self.file.write("\n")
            self.file.flush()

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("Progress is not iterable without an iterable")
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()

    def __enter__(self) -> Progress[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def progress(
    iterable: Iterable[T] | None = None,
    *,
    total: float | None = None,
    desc: str | None = None,
    unit: str = "it",
    initial: float = 0,
    mininterval: float = 0.5,
    file: TextIO | None = None,
    disable: bool | None = None,
) -> Progress[T]:
    """Create a cloud-observable progress tracker with tqdm-style semantics."""

    return Progress(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        initial=initial,
        mininterval=mininterval,
        file=file,
        disable=disable,
    )


__all__ = ["Progress", "progress"]
