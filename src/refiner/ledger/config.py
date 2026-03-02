from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .backend.base import LedgerConfig


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Invalid int env var {name}={v!r}") from e


def load_ledger_config_from_env(default: LedgerConfig | None = None) -> LedgerConfig:
    cfg = default or LedgerConfig()
    lease = _env_int("REFINER_LEDGER_LEASE_SECONDS", cfg.lease_seconds)
    hb = _env_int("REFINER_LEDGER_HEARTBEAT_SECONDS", cfg.heartbeat_seconds)
    return replace(cfg, lease_seconds=int(lease), heartbeat_seconds=int(hb))


def default_workdir() -> str:
    """Return the default Refiner working directory root (cache-backed)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return str(Path(xdg) / "macrodata" / "refiner")
    return str(Path.home() / ".cache" / "macrodata" / "refiner")


def resolve_workdir(workdir: str | None) -> str:
    """Resolve/validate the workdir (must be absolute)."""
    if workdir is None or workdir == "":
        workdir = os.environ.get("REFINER_WORKDIR") or default_workdir()
    p = Path(os.path.expanduser(workdir))
    if not p.is_absolute():
        raise ValueError("REFINER_WORKDIR must be an absolute path")
    return str(p)


__all__ = [
    "default_workdir",
    "resolve_workdir",
    "load_ledger_config_from_env",
]
