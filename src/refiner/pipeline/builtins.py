"""Lightweight callable descriptions, usable without importing the planner."""

from typing import Any


def describe_builtin(
    name: str, *, refiner_extras: tuple[str, ...] = (), **args: Any
) -> Any:
    def _decorate(fn: Any) -> Any:
        setattr(
            fn,
            "__refiner_builtin_call__",
            {
                "name": name,
                "args": args,
                "services": (),
                "refiner_extras": refiner_extras,
            },
        )
        return fn

    return _decorate
