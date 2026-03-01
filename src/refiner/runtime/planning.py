from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


def _step_name_type(step: Any) -> tuple[str, str, dict[str, Any] | None]:
    from refiner.processors.step import FnBatchStep, FnFlatMapStep, FnRowStep

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
    reader_name = pipeline.source.__class__.__name__.replace("Reader", "").lower()
    source_step_name = f"read_{reader_name}"
    files: list[Any] = list(getattr(pipeline.source, "files", []))
    path_arg = files[0] if files else ""
    if len(files) > 1 and path_arg:
        path_arg = f"{path_arg} (+{len(files) - 1} more)"
    source_args: dict[str, Any] = {}
    if path_arg:
        source_args["path"] = path_arg

    steps: list[dict[str, Any]] = []
    used_names: dict[str, int] = {}

    def _unique_name(base: str) -> str:
        count = used_names.get(base, 0) + 1
        used_names[base] = count
        return base if count == 1 else f"{base}_{count}"

    steps.append(
        _step_payload(
            name=_unique_name(source_step_name),
            step_type="reader",
            index=0,
            args=source_args,
            code=None,
        )
    )
    for step in pipeline.pipeline_steps:
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
