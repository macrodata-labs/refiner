from __future__ import annotations

from collections.abc import Collection, Mapping

from refiner.inference.types import InferenceWarning, ProviderOptions


def provider_option_warnings(
    *,
    provider_name: str,
    expected_namespace: str | Collection[str],
    supported_options: Collection[str],
    provider_options: ProviderOptions | None,
) -> list[InferenceWarning]:
    if not provider_options:
        return []
    expected_namespaces = _expected_namespaces(expected_namespace)
    warnings: list[InferenceWarning] = []
    for namespace in provider_options:
        if namespace not in expected_namespaces:
            warnings.append(
                {
                    "type": "unsupported-provider-option",
                    "setting": f"provider_options.{namespace}",
                    "message": (
                        f"{namespace!r} provider options are not used by "
                        f"{provider_name}."
                    ),
                }
            )
    for namespace in expected_namespaces:
        expected_options = provider_options.get(namespace, {})
        if isinstance(expected_options, Mapping):
            for option in expected_options:
                if option not in supported_options:
                    warnings.append(
                        {
                            "type": "unsupported-setting",
                            "setting": f"provider_options.{namespace}.{option}",
                            "message": (
                                f"{option!r} is not currently mapped by "
                                f"{provider_name}."
                            ),
                        }
                    )
    return warnings


def _expected_namespaces(namespace: str | Collection[str]) -> tuple[str, ...]:
    if isinstance(namespace, str):
        return (namespace,)
    return tuple(namespace)


__all__ = ["provider_option_warnings"]
