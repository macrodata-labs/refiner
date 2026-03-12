from __future__ import annotations

import os
from pathlib import Path


def default_workdir() -> str:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return str(Path(xdg) / "macrodata" / "refiner")
    return str(Path.home() / ".cache" / "macrodata" / "refiner")


def resolve_workdir(workdir: str | None) -> str:
    value = workdir or os.environ.get("REFINER_WORKDIR") or default_workdir()
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        raise ValueError("REFINER_WORKDIR must be an absolute path")
    return str(path)


__all__ = ["default_workdir", "resolve_workdir"]
