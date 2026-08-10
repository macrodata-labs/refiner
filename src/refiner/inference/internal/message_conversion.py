from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def _custom_provider_data(
    part: Mapping[str, Any], provider_aliases: set[str]
) -> dict[str, Any]:
    provider = part.get("provider")
    if not isinstance(provider, str) or provider not in provider_aliases:
        aliases = ", ".join(sorted(provider_aliases))
        raise ValueError(
            f"custom content part provider must be one of {aliases}; got {provider!r}"
        )
    data = part.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("custom content part data must be an object")
    return dict(data)


def _provider_option(part: Mapping[str, Any], provider: str, key: str) -> object:
    provider_options = part.get("provider_options")
    if not isinstance(provider_options, Mapping):
        return None
    options = cast(Mapping[str, Any], provider_options).get(provider)
    if not isinstance(options, Mapping):
        return None
    return options.get(key)


def _provider_options(provider_options: object, provider: str) -> Mapping[str, Any]:
    if not isinstance(provider_options, Mapping):
        return {}
    options = dict(provider_options).get(provider)
    if not isinstance(options, Mapping):
        return {}
    return options


__all__ = [
    "_custom_provider_data",
    "_provider_option",
    "_provider_options",
]
