from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


def _step_name_type(step: Any) -> tuple[str, str, dict[str, Any] | None]:
    from refiner.processors.step import (
        CastStep,
        DropStep,
        FilterExprStep,
        FnBatchStep,
        FnFlatMapStep,
        FnRowStep,
        RenameStep,
        SelectStep,
        WithColumnsStep,
    )

    explicit_name = getattr(step, "op_name", None)
    if isinstance(step, FnRowStep):
        return (explicit_name or "map"), "row_map", None
    if isinstance(step, FnBatchStep):
        return (
            (explicit_name or "batch_map"),
            "batch_map",
            {"batch_size": step.batch_size},
        )
    if isinstance(step, FnFlatMapStep):
        return (explicit_name or "flat_map"), "flat_map", None
    if isinstance(step, SelectStep):
        return (explicit_name or "select"), "select", {"columns": list(step.columns)}
    if isinstance(step, WithColumnsStep):
        return (
            explicit_name or "with_columns",
            "with_columns",
            {"columns": {k: v.to_plan() for k, v in step.assignments.items()}},
        )
    if isinstance(step, DropStep):
        return (explicit_name or "drop"), "drop", {"columns": list(step.columns)}
    if isinstance(step, RenameStep):
        return (explicit_name or "rename"), "rename", {"mapping": dict(step.mapping)}
    if isinstance(step, CastStep):
        return (explicit_name or "cast"), "cast", {"dtypes": dict(step.dtypes)}
    if isinstance(step, FilterExprStep):
        return (
            explicit_name or "filter",
            "filter_expr",
            {"predicate": step.predicate.to_plan()},
        )
    return step.__class__.__name__, step.__class__.__name__.lower(), None


def _callable_code_hint(obj: Any) -> str | None:
    fn = getattr(obj, "fn", None)
    if fn is None:
        return None
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return repr(fn)


def _step_payload(
    *,
    name: str,
    step_type: str,
    index: int,
    args: dict[str, Any] | None,
    code: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "type": step_type, "index": index}
    if args:
        payload["args"] = args
    if code is not None:
        payload["code"] = code
    return payload


def compile_pipeline_plan(pipeline: "RefinerPipeline") -> dict[str, Any]:
    """Compile a transport-neutral plan description for a pipeline."""
    source_step_name = str(getattr(pipeline.source, "name", "source"))
    source_args: dict[str, Any] = dict(pipeline.source.describe())

    steps: list[dict[str, Any]] = []
    used_names: dict[str, int] = {}

    def _unique_name(base: str) -> str:
        count = used_names.get(base, 0) + 1
        used_names[base] = count
        return base if count == 1 else f"{base}_{count}"

    steps.append(
        _step_payload(
            name=_unique_name(source_step_name),
            step_type="source",
            index=0,
            args=source_args,
            code=None,
        )
    )
    from refiner.processors.step import VectorizedSegmentStep

    for step in pipeline.pipeline_steps:
        if isinstance(step, VectorizedSegmentStep):
            for op in step.ops:
                base_name, step_type, args = _step_name_type(op)
                steps.append(
                    _step_payload(
                        name=_unique_name(base_name),
                        step_type=step_type,
                        args=args,
                        code=None,
                    )
                )
            continue
        base_name, step_type, args = _step_name_type(step)
        steps.append(
            _step_payload(
                name=_unique_name(base_name),
                step_type=step_type,
                index=step.index,
                args=args,
                code=_callable_code_hint(step),
            )
        )

    return {
        "stages": [
            {
                "name": "stage_0",
                "index": 0,
                "steps": steps,
            }
        ]
    }


__all__ = ["compile_pipeline_plan"]
