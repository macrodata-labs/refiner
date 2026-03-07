from __future__ import annotations

import ast
import inspect
import textwrap
from types import CodeType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


def _explicit_callable_name(fn: Any) -> str | None:
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    if not normalized or normalized == "<lambda>":
        return None
    return normalized


def _step_name_type(step: Any) -> tuple[str, str, dict[str, Any] | None]:
    from refiner.processors.step import (
        CastStep,
        DropStep,
        FilterExprStep,
        FilterRowStep,
        FnBatchStep,
        FnFlatMapStep,
        FnRowStep,
        RenameStep,
        SelectStep,
        WithColumnsStep,
    )

    explicit_name = getattr(step, "op_name", None)
    if isinstance(step, FnRowStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "map"
            else _explicit_callable_name(step.fn)
        )
        return (inferred_name or "map"), "row_map", {"fn": step.fn}
    if isinstance(step, FnBatchStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "batch_map"
            else _explicit_callable_name(step.fn)
        )
        return (
            (inferred_name or "batch_map"),
            "batch_map",
            {"fn": step.fn, "batch_size": step.batch_size},
        )
    if isinstance(step, FilterRowStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "filter"
            else _explicit_callable_name(step.predicate)
        )
        return (inferred_name or "filter"), "filter", {"fn": step.predicate}
    if isinstance(step, FnFlatMapStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "flat_map"
            else _explicit_callable_name(step.fn)
        )
        step_name = inferred_name or "flat_map"
        return step_name, "flat_map", {"fn": step.fn}
    if isinstance(step, SelectStep):
        return (explicit_name or "select"), "select", {"columns": list(step.columns)}
    if isinstance(step, WithColumnsStep):
        return (
            explicit_name or "with_columns",
            "with_columns",
            {name: expr.to_code() for name, expr in step.assignments.items()},
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
            {"expression": step.predicate.to_code()},
        )
    return step.__class__.__name__, step.__class__.__name__.lower(), None


def _parse_lambda_segments(source: str) -> list[str]:
    attempts = [source]
    if source.startswith("."):
        # inspect.getsource can return chained call fragments like
        # ".filter(lambda row: ...)", which are invalid as standalone syntax.
        attempts.append(f"_refiner_receiver{source}")

    segments: list[tuple[int, int, str]] = []
    seen_segments: set[str] = set()
    for candidate in attempts:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Lambda):
                continue
            segment = ast.get_source_segment(candidate, node)
            if not isinstance(segment, str):
                continue
            normalized = segment.strip()
            if not normalized or normalized in seen_segments:
                continue
            seen_segments.add(normalized)
            segments.append((node.lineno, node.col_offset, normalized))

    segments.sort(key=lambda item: (item[0], item[1]))
    return [segment for _, _, segment in segments]


def _compiled_lambda_code(source: str) -> CodeType | None:
    try:
        code = compile(source, "<refiner-lambda>", "eval")
    except SyntaxError:
        return None

    for const in code.co_consts:
        if isinstance(const, CodeType) and const.co_name == "<lambda>":
            return const
    return None


def _const_fingerprint(value: Any) -> Any:
    if isinstance(value, CodeType):
        return _code_fingerprint(value)
    return value


def _code_fingerprint(code: CodeType) -> tuple[Any, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        tuple(_const_fingerprint(const) for const in code.co_consts),
    )


def _code_objects_equal(left: CodeType, right: CodeType) -> bool:
    return _code_fingerprint(left) == _code_fingerprint(right)


def _extract_lambda_source(source: str, fn: Any) -> str | None:
    segments = _parse_lambda_segments(source)
    if not segments:
        return None

    target_code = getattr(fn, "__code__", None)
    if isinstance(target_code, CodeType):
        for segment in segments:
            candidate = _compiled_lambda_code(segment)
            if isinstance(candidate, CodeType) and _code_objects_equal(
                candidate, target_code
            ):
                return segment

    return segments[0]


def _callable_source(fn: Any) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = None

    if isinstance(source, str) and source.strip():
        normalized = textwrap.dedent(source).strip()
        lambda_source = _extract_lambda_source(normalized, fn)
        return lambda_source or normalized

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
) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "type": step_type, "index": index}
    if args:
        payload["args"] = args
    return payload


def _serialize_args(args: dict[str, Any] | None) -> dict[str, Any] | None:
    if not args:
        return args

    serialized: dict[str, Any] = {}
    meta: dict[str, str] = {}
    for key, value in args.items():
        if callable(value):
            source = _callable_source(value)
            if source is not None:
                serialized[key] = source
                meta[key] = "code"
                continue
        serialized[key] = value

    if meta:
        serialized["__meta"] = meta
    return serialized


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

    source_name = _unique_name(source_step_name)
    steps.append(
        _step_payload(
            name=source_name,
            step_type="source",
            index=0,
            args=_serialize_args(source_args),
        )
    )
    from refiner.processors.step import VectorizedSegmentStep

    for step in pipeline.pipeline_steps:
        if isinstance(step, VectorizedSegmentStep):
            for op in step.ops:
                base_name, step_type, args = _step_name_type(op)
                unique_name = _unique_name(base_name)
                steps.append(
                    _step_payload(
                        name=unique_name,
                        step_type=step_type,
                        index=len(steps),
                        args=_serialize_args(args),
                    )
                )
            continue
        base_name, step_type, args = _step_name_type(step)
        unique_name = _unique_name(base_name)
        steps.append(
            _step_payload(
                name=unique_name,
                step_type=step_type,
                index=step.index,
                args=_serialize_args(args),
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
