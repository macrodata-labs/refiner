from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from refiner.services.base import (
    BaseGenerationService,
    RuntimeServiceBinding,
    RuntimeServiceDefinition,
)


@dataclass(slots=True)
class ServiceRegistry:
    services: Mapping[str, BaseGenerationService]

    @classmethod
    def from_definitions(
        cls,
        *,
        definitions: Sequence[RuntimeServiceDefinition],
        bindings: Sequence[RuntimeServiceBinding] = (),
    ) -> ServiceRegistry:
        bindings_by_name: dict[str, RuntimeServiceBinding] = {}
        for binding in bindings:
            if binding.name in bindings_by_name:
                raise ValueError(f"duplicate service binding name {binding.name!r}")
            bindings_by_name[binding.name] = binding

        clients: dict[str, BaseGenerationService] = {}
        seen_names: dict[str, RuntimeServiceDefinition] = {}
        for definition in definitions:
            previous = seen_names.get(definition.name)
            if previous is not None and previous != definition:
                raise ValueError(
                    f"duplicate service definition name {definition.name!r} with mismatched configuration"
                )
            seen_names[definition.name] = definition
            clients[definition.name] = definition.build_client(
                bindings_by_name.get(definition.name)
            )
        return cls(services=clients)

    def get(self, name: str) -> BaseGenerationService:
        try:
            return self.services[name]
        except KeyError as err:
            raise KeyError(f"unknown service {name!r}") from err


__all__ = ["ServiceRegistry"]
