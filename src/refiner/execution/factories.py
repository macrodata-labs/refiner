from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from refiner.execution.operators.row import close_async_steps
from refiner.pipeline.steps import (
    FnAsyncBatchStep,
    FnAsyncRowStep,
    FnBatchStep,
    FnTableStep,
    RefinerStep,
    VectorizedSegmentStep,
)
from refiner.worker.context import logger, set_active_step_index

FactoryStep = FnAsyncRowStep | FnAsyncBatchStep | FnBatchStep | FnTableStep


def _factory_step(step: RefinerStep) -> FactoryStep | None:
    if isinstance(step, FactoryStep):
        return step
    return None


def _step_has_factory(step: RefinerStep) -> bool:
    if isinstance(step, VectorizedSegmentStep):
        return any(
            op.factory is not None for op in step.ops if isinstance(op, FnTableStep)
        )
    factory_step = _factory_step(step)
    return factory_step is not None and factory_step.factory is not None


def has_worker_factories(steps: Sequence[RefinerStep]) -> bool:
    """Return whether any step needs worker-local callable initialization."""
    return any(_step_has_factory(step) for step in steps)


def _initialize_factory_step(step: FactoryStep) -> FactoryStep:
    factory = step.factory
    if factory is None:
        return step
    with set_active_step_index(step.index):
        fn: Any = factory()
    if not callable(fn):
        op_name = step.op_name or type(step).__name__
        raise TypeError(f"{op_name} factory must return a callable, got {type(fn)!r}")
    return replace(step, fn=fn, factory=None)


def initialize_worker_steps(
    steps: Sequence[RefinerStep],
) -> tuple[RefinerStep, ...]:
    """Instantiate each configured factory once for one worker execution."""
    initialized: list[RefinerStep] = []
    try:
        for step in steps:
            if isinstance(step, VectorizedSegmentStep):
                initialized.append(
                    replace(
                        step,
                        ops=tuple(
                            _initialize_factory_step(op)
                            if isinstance(op, FnTableStep)
                            else op
                            for op in step.ops
                        ),
                    )
                )
                continue
            factory_step = _factory_step(step)
            initialized.append(
                _initialize_factory_step(factory_step)
                if factory_step is not None
                else step
            )
    except BaseException as initialization_error:
        try:
            close_async_steps(initialized)
        except Exception as teardown_error:  # noqa: BLE001
            logger.warning(
                "factory cleanup failed after initialization error: {}: {}",
                type(teardown_error).__name__,
                teardown_error,
            )
            add_note = getattr(initialization_error, "add_note", None)
            if callable(add_note):
                add_note(
                    "Factory cleanup also failed: "
                    f"{type(teardown_error).__name__}: {teardown_error}"
                )
        raise
    return tuple(initialized)


__all__ = [
    "has_worker_factories",
    "initialize_worker_steps",
]
