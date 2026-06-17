from __future__ import annotations

from typing import Any

from refiner.services.base import RuntimeServiceSpec

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


def describe_builtin(
    name: str, *, refiner_extras: tuple[str, ...] = (), **args: Any
) -> Any:
    def _decorate(fn: Any) -> Any:
        setattr(
            fn,
            _REFINER_BUILTIN_CALL_ATTR,
            {
                "name": name,
                "args": args,
                "services": (),
                "refiner_extras": refiner_extras,
            },
        )
        return fn

    return _decorate


__all__ = [
    "_REFINER_BUILTIN_CALL_ATTR",
    "_builtin_description",
    "describe_builtin",
]
