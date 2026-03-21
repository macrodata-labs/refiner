from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from types import CodeType
from typing import TYPE_CHECKING, Any

from refiner.platform.manifest import _redact_captured_text

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"


@dataclass(frozen=True, slots=True)
class StageComputeRequirements:
    num_workers: int


@dataclass(frozen=True, slots=True)
class PlannedStage:
    index: int
    name: str
    pipeline: "RefinerPipeline"
    compute: StageComputeRequirements


def _explicit_callable_name(fn: Any) -> str | None:
    builtin_description = _builtin_description(fn)
    if builtin_description is not None:
        return builtin_description["name"]
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str):
        return None
    normalized = name.strip()
    if not normalized or normalized == "<lambda>":
        return None
    return normalized


def _callable_step_args(
    fn: Any,
    *,
    extra_args: dict[str, Any] | None = None,
    builtin_extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    builtin_description = _builtin_description(fn)
    if builtin_description is None:
        args: dict[str, Any] = {"fn": fn}
        if extra_args:
            args.update(extra_args)
    else:
        args = dict(builtin_description["args"])
        if builtin_extra_args:
            args.update(builtin_extra_args)
    return args


def _step_name_type(step: Any) -> tuple[str, str, dict[str, Any] | None]:
    from refiner.pipeline.steps import (
        CastStep,
        DropStep,
        FilterExprStep,
        FilterRowStep,
        FnAsyncRowStep,
        FnBatchStep,
        FnFlatMapStep,
        FnRowStep,
        FnTableStep,
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
        return (
            (inferred_name or "map"),
            "row_map",
            _callable_step_args(step.fn),
        )
    if isinstance(step, FnAsyncRowStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "map_async"
            else _explicit_callable_name(step.fn)
        )
        return (
            (inferred_name or "map_async"),
            "async_map",
            _callable_step_args(
                step.fn,
                extra_args={
                    "max_in_flight": step.max_in_flight,
                    "preserve_order": step.preserve_order,
                },
                builtin_extra_args={
                    "async_map.max_in_flight": step.max_in_flight,
                    "async_map.preserve_order": step.preserve_order,
                },
            ),
        )
    if isinstance(step, FnBatchStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "batch_map"
            else _explicit_callable_name(step.fn)
        )
        return (
            (inferred_name or "batch_map"),
            "batch_map",
            _callable_step_args(
                step.fn,
                extra_args={"batch_size": step.batch_size},
                builtin_extra_args={"batch_map.batch_size": step.batch_size},
            ),
        )
    if isinstance(step, FilterRowStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "filter"
            else _explicit_callable_name(step.predicate)
        )
        return (
            (inferred_name or "filter"),
            "filter",
            _callable_step_args(step.predicate),
        )
    if isinstance(step, FnFlatMapStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "flat_map"
            else _explicit_callable_name(step.fn)
        )
        step_name = inferred_name or "flat_map"
        return step_name, "flat_map", _callable_step_args(step.fn)
    if isinstance(step, FnTableStep):
        inferred_name = (
            explicit_name
            if explicit_name and explicit_name != "map_table"
            else _explicit_callable_name(step.fn)
        )
        return (
            (inferred_name or "map_table"),
            "table_map",
            _callable_step_args(step.fn),
        )
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


def _builtin_description(fn: Any) -> dict[str, Any] | None:
    spec = getattr(fn, _REFINER_BUILTIN_CALL_ATTR, None)
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = spec.get("args")
    if not isinstance(args, dict):
        return None
    return {"name": name, "args": args}


def describe_builtin(name: str, **args: Any) -> Any:
    def _decorate(fn: Any) -> Any:
        setattr(fn, _REFINER_BUILTIN_CALL_ATTR, {"name": name, "args": args})
        return fn

    return _decorate


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


def _sink_name_type(sink: Any) -> tuple[str, str, dict[str, Any] | None]:
    payload = sink.describe()
    if payload is not None:
        return payload
    sink_name = sink.__class__.__name__.replace("Sink", "").lower()
    return sink_name or "sink", "writer", None


def _serialize_args(
    args: dict[str, Any] | None, *, secret_values: tuple[str, ...] = ()
) -> dict[str, Any] | None:
    if not args:
        return args

    serialized: dict[str, Any] = {}
    meta: dict[str, str] = {}
    for key, value in args.items():
        if callable(value):
            source = _callable_source(value)
            if source is not None:
                serialized[key] = (
                    _redact_captured_text(source, secret_values=secret_values)
                    if secret_values
                    else source
                )
                meta[key] = "code"
                continue
        serialized[key] = value

    if meta:
        serialized["__meta"] = meta
    return serialized


def _compile_stage_steps(
    pipeline: "RefinerPipeline", *, secret_values: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
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
            args=_serialize_args(source_args, secret_values=secret_values),
        )
    )
    from refiner.pipeline.steps import VectorizedSegmentStep

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
                        args=_serialize_args(args, secret_values=secret_values),
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
                args=_serialize_args(args, secret_values=secret_values),
            )
        )

    if pipeline.sink is not None:
        base_name, step_type, args = _sink_name_type(pipeline.sink)
        unique_name = _unique_name(base_name)
        steps.append(
            _step_payload(
                name=unique_name,
                step_type=step_type,
                index=len(steps),
                args=_serialize_args(args, secret_values=secret_values),
            )
        )

    return steps


def plan_pipeline_stages(
    pipeline: "RefinerPipeline", *, default_num_workers: int
) -> list[PlannedStage]:
    """Return the ordered execution stages for a pipeline.

    This is currently a placeholder splitter that yields a single stage. Future
    multi-stage planning logic should live here.
    """
    if default_num_workers <= 0:
        raise ValueError("default_num_workers must be > 0")

    from refiner.pipeline.pipeline import RefinerPipeline
    from refiner.pipeline.sinks.lerobot import LeRobotWriterSink
    from refiner.pipeline.sinks.lerobot_reducer import LeRobotMetaReduceSink
    from refiner.pipeline.sources.task import TaskSource

    if isinstance(pipeline.sink, LeRobotWriterSink):
        reducer_stage = RefinerPipeline(
            source=TaskSource(num_tasks=1),
            pipeline_steps=(),
            max_vectorized_block_bytes=pipeline.max_vectorized_block_bytes,
            sink=LeRobotMetaReduceSink(output=pipeline.sink.output),
        )
        return [
            PlannedStage(
                index=0,
                name="write_lerobot_stage_1",
                pipeline=pipeline,
                compute=StageComputeRequirements(num_workers=default_num_workers),
            ),
            PlannedStage(
                index=1,
                name="write_lerobot_stage_2",
                pipeline=reducer_stage,
                compute=StageComputeRequirements(num_workers=1),
            ),
        ]

    return [
        PlannedStage(
            index=0,
            name="stage_0",
            pipeline=pipeline,
            compute=StageComputeRequirements(num_workers=default_num_workers),
        )
    ]


def compile_planned_stages(
    stages: list[PlannedStage], *, secret_values: tuple[str, ...] = ()
) -> dict[str, Any]:
    plan = {
        "stages": [
            {
                "name": stage.name,
                "index": stage.index,
                "requested_num_workers": stage.compute.num_workers,
                "steps": _compile_stage_steps(
                    stage.pipeline, secret_values=secret_values
                ),
            }
            for stage in stages
        ]
    }
    return plan


def compile_pipeline_plan(
    pipeline: "RefinerPipeline", *, secret_values: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Compile a transport-neutral single-pipeline plan description."""
    return compile_planned_stages(
        plan_pipeline_stages(pipeline, default_num_workers=1),
        secret_values=secret_values,
    )


__all__ = [
    "PlannedStage",
    "StageComputeRequirements",
    "compile_pipeline_plan",
    "compile_planned_stages",
    "plan_pipeline_stages",
]
