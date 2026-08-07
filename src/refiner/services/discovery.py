from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from refiner.pipeline.builtins import iter_pipeline_builtin_specs
from refiner.services.base import RuntimeServiceSpec

if TYPE_CHECKING:
    from refiner.pipeline import RefinerPipeline


def collect_pipeline_services(
    pipeline: "RefinerPipeline",
) -> tuple[RuntimeServiceSpec, ...]:
    services_by_key: dict[tuple[str, str, str], RuntimeServiceSpec] = {}

    for spec in iter_pipeline_builtin_specs(pipeline):
        for service in spec.services:
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
