from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from refiner.pipeline.steps import (
    FnAsyncBatchStep,
    FnAsyncRowStep,
    FnBatchStep,
    FnFlatMapStep,
    FnRowStep,
    FnTableStep,
    VectorizedSegmentStep,
    ValidationStep,
)
from refiner.services.base import RuntimeServiceSpec

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


_REFINER_BUILTIN_CALL_ATTR = "__refiner_builtin_call__"


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
    services = spec.get("services", ())
    if not isinstance(services, (list, tuple)):
        return None
    parsed_services: list[RuntimeServiceSpec] = []
    for service in services:
        if not isinstance(service, RuntimeServiceSpec):
            return None
        parsed_services.append(service)
    return {"name": name, "args": args, "services": tuple(parsed_services)}


def collect_pipeline_services(
    pipeline: "RefinerPipeline",
) -> tuple[RuntimeServiceSpec, ...]:
    services_by_key: dict[tuple[str, str, str], RuntimeServiceSpec] = {}

    for step in pipeline.pipeline_steps:
        candidates: list[Any] = []
        callable_steps = (
            step.ops if isinstance(step, VectorizedSegmentStep) else (step,)
        )
        for callable_step in callable_steps:
            if isinstance(
                callable_step,
                FnRowStep
                | FnAsyncRowStep
                | FnAsyncBatchStep
                | FnBatchStep
                | FnFlatMapStep
                | FnTableStep,
            ):
                factory = getattr(callable_step, "factory", None)
                candidate = (
                    factory
                    if factory is not None
                    else getattr(callable_step, "fn", None)
                )
                if candidate is not None:
                    candidates.append(candidate)
            elif (fn := getattr(callable_step, "fn", None)) is not None:
                candidates.append(fn)
            elif isinstance(callable_step, ValidationStep):
                candidates.extend(callable_step.contract.predicates.values())

        for candidate in candidates:
            builtin = _builtin_description(candidate)
            if builtin is None:
                continue
            for service in builtin["services"]:
                key = (
                    service.name,
                    service.kind,
                    _service_config_key(service.config),
                )
                services_by_key.setdefault(key, service)
    return tuple(services_by_key.values())


def runtime_service_specs_to_dicts(
    services: Sequence[RuntimeServiceSpec],
) -> list[dict[str, Any]]:
    return [service.to_dict() for service in services]


def parse_runtime_service_specs(
    services: Sequence[Mapping[str, Any]],
) -> tuple[RuntimeServiceSpec, ...]:
    return tuple(RuntimeServiceSpec.from_dict(service) for service in services)


__all__ = [
    "collect_pipeline_services",
    "parse_runtime_service_specs",
    "runtime_service_specs_to_dicts",
]


def _service_config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(
        _jsonify_config_value(config),
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonify_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonify_config_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonify_config_value(item) for item in value]
    return value
