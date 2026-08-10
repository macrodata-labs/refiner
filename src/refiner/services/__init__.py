from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY_ATTRS = {
    "RuntimeServiceBinding": "refiner.services.base",
    "RuntimeServiceSpec": "refiner.services.base",
    "ServiceManager": "refiner.services.manager",
    "VLLMRuntimeServiceBinding": "refiner.services.vllm",
    "VLLMServiceDefinition": "refiner.services.vllm",
}

__all__ = [
    "RuntimeServiceBinding",
    "RuntimeServiceSpec",
    "ServiceManager",
    "VLLMServiceDefinition",
    "VLLMRuntimeServiceBinding",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_ATTRS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from refiner.services.base import RuntimeServiceBinding, RuntimeServiceSpec
    from refiner.services.manager import ServiceManager
    from refiner.services.vllm import VLLMRuntimeServiceBinding, VLLMServiceDefinition
