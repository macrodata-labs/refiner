from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from refiner.pipeline.steps import (
    FnAsyncRowStep,
    FnBatchStep,
    FnFlatMapStep,
    FnRowStep,
    FnTableStep,
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
    services_by_key: dict[
        tuple[str, str, tuple[tuple[str, Any], ...]], RuntimeServiceSpec
    ] = {}

    for step in pipeline.pipeline_steps:
        candidates: list[Any] = []
        if isinstance(
            step, FnRowStep | FnAsyncRowStep | FnBatchStep | FnFlatMapStep | FnTableStep
        ):
            candidates.append(step.fn)

        for candidate in candidates:
            builtin = _builtin_description(candidate)
            if builtin is None:
                continue
            for service in builtin["services"]:
                key = (
                    service.name,
                    service.kind,
                    tuple(
                        sorted(
                            (str(k), _freeze_config_value(v))
                            for k, v in service.config.items()
                        )
                    ),
                )
                services_by_key.setdefault(key, service)
    return tuple(services_by_key.values())


__all__ = ["collect_pipeline_services"]


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_config_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_config_value(item) for item in value)
    return value
