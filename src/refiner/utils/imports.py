from __future__ import annotations

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import NoReturn


DependencySpec = str | tuple[str, str]


def check_required_dependencies(
    step_name: str,
    required_dependencies: list[DependencySpec] | tuple[DependencySpec, ...],
    *,
    dist: str | None = None,
) -> None:
    missing_dependencies: dict[str, str] = {}
    for dependency in required_dependencies:
        package_name, pip_name = (
            dependency if isinstance(dependency, tuple) else (dependency, dependency)
        )
        if not _is_package_available(package_name):
            missing_dependencies[package_name] = pip_name
        if not _is_distribution_available(pip_name):
            missing_dependencies[package_name] = pip_name
    if missing_dependencies:
        _raise_error_for_missing_dependencies(
            step_name=step_name,
            dependencies=missing_dependencies,
            dist=dist,
        )


def _raise_error_for_missing_dependencies(
    *,
    step_name: str,
    dependencies: dict[str, str],
    dist: str | None,
) -> NoReturn:
    dependencies = dict(sorted(dependencies.items()))
    package_names = list(dependencies)
    if len(package_names) > 1:
        packages = (
            f"{','.join('`' + name + '`' for name in package_names[:-1])} "
            f"and `{package_names[-1]}`"
        )
    else:
        packages = f"`{package_names[0]}`"
    pip_packages: list[str] = []
    for pip_name in dependencies.values():
        if pip_name not in pip_packages:
            pip_packages.append(pip_name)
    package_command = f"`pip install {' '.join(pip_packages)}`"
    message = f"Please install {packages} to use {step_name} ({package_command})."
    if dist is not None:
        message = f"{message[:-1]}, or simply `pip install macrodata-refiner[{dist}]`."
    raise ImportError(message)


@lru_cache
def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


@lru_cache
def _is_distribution_available(distribution_name: str) -> bool:
    try:
        importlib.metadata.distribution(distribution_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


__all__ = ["check_required_dependencies"]
