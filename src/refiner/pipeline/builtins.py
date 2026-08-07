from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from refiner.pipeline.steps import VectorizedSegmentStep
from refiner.services.base import RuntimeServiceSpec

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline

REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"


@dataclass(frozen=True, slots=True)
class BuiltinCallSpec:
    name: str
    args: dict[str, Any]
    services: tuple[RuntimeServiceSpec, ...] = ()
    refiner_extras: tuple[str, ...] = ()


def builtin_call_spec(fn: Any) -> BuiltinCallSpec | None:
    spec = getattr(fn, REFINER_BUILTIN_CALL_ATTR, None)
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = spec.get("args")
    if not isinstance(args, dict):
        return None
    services = spec.get("services", ())
    if not isinstance(services, (list, tuple)):
        return None
    parsed_services: list[RuntimeServiceSpec] = []
    for service in services:
        if not isinstance(service, RuntimeServiceSpec):
            return None
        parsed_services.append(service)
    refiner_extras = spec.get("refiner_extras", ())
    if not isinstance(refiner_extras, tuple) or not all(
        isinstance(extra, str) for extra in refiner_extras
    ):
        return None
    return BuiltinCallSpec(
        name=name,
        args=args,
        services=tuple(parsed_services),
        refiner_extras=refiner_extras,
    )


def iter_pipeline_builtin_specs(
    pipeline: "RefinerPipeline",
) -> Iterator[BuiltinCallSpec]:
    seen: set[int] = set()
    for step in pipeline.pipeline_steps:
        candidates = step.ops if isinstance(step, VectorizedSegmentStep) else (step,)
        for candidate in candidates:
            for attr in ("fn", "predicate"):
                fn = getattr(candidate, attr, None)
                if fn is None or id(fn) in seen:
                    continue
                seen.add(id(fn))
                if spec := builtin_call_spec(fn):
                    yield spec


__all__ = [
    "BuiltinCallSpec",
    "REFINER_BUILTIN_CALL_ATTR",
    "builtin_call_spec",
    "iter_pipeline_builtin_specs",
]
