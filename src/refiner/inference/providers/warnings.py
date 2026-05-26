from __future__ import annotations

from collections.abc import Collection

from refiner.inference.types import InferenceWarning, ProviderOptions


def provider_option_warnings(
    *,
    provider_name: str,
    expected_namespace: str,
    supported_options: Collection[str],
    provider_options: ProviderOptions | None,
) -> list[InferenceWarning]:
    if not provider_options:
        return []
    warnings: list[InferenceWarning] = []
    for namespace in provider_options:
        if namespace != expected_namespace:
            warnings.append(
                {
                    "type": "unsupported-provider-option",
                    "setting": f"providerOptions.{namespace}",
                    "message": (
                        f"{namespace!r} provider options are not used by "
                        f"{provider_name}."
                    ),
                }
            )
    expected_options = provider_options.get(expected_namespace, {})
    for option in expected_options:
        if option not in supported_options:
            warnings.append(
                {
                    "type": "unsupported-setting",
                    "setting": f"providerOptions.{expected_namespace}.{option}",
                    "message": (
                        f"{option!r} is not currently mapped by {provider_name}."
                    ),
                }
            )
    return warnings


__all__ = ["provider_option_warnings"]
