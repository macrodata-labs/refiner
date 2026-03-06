from __future__ import annotations

import hashlib
import inspect
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def _is_external_script(path: Path) -> bool:
    resolved_str = str(path)
    if "/site-packages/" in resolved_str or "/dist-packages/" in resolved_str:
        return False
    if "/src/refiner/" in resolved_str:
        return False
    return True


def _detect_script_path() -> Path | None:
    argv0 = sys.argv[0].strip() if sys.argv else ""
    if argv0 and argv0 not in {"-", "-c"}:
        argv_path = Path(argv0).expanduser()
        if argv_path.is_file():
            return argv_path.resolve()

    for frame in inspect.stack():
        filename = frame.filename
        if not filename:
            continue
        frame_path = Path(filename)
        if not frame_path.is_file():
            continue
        resolved = frame_path.resolve()
        if not _is_external_script(resolved):
            continue
        return resolved
    return None


def _read_script(
    script_path: Path | None,
) -> tuple[str | None, str | None, str | None]:
    if script_path is None:
        return None, None, None

    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return str(script_path), None, None

    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(script_path), text, sha256


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _collect_dependencies() -> list[dict[str, str]]:
    dependencies_by_name: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        dist_name_raw = dist.metadata["Name"]
        dist_name = str(dist_name_raw).strip() if dist_name_raw else ""
        if not dist_name:
            continue
        dist_version = dist.version
        if not isinstance(dist_version, str) or not dist_version:
            continue
        dependencies_by_name[dist_name] = dist_version

    return [
        {"name": name, "version": dependencies_by_name[name]}
        for name in sorted(dependencies_by_name, key=lambda item: item.lower())
    ]


def build_run_manifest() -> dict[str, Any]:
    script_path = _detect_script_path()
    path, text, sha256 = _read_script(script_path)

    refiner_version = _package_version("refiner")
    if refiner_version is None:
        refiner_version = _package_version("macrodata-refiner")

    return {
        "version": 1,
        "script": {
            "path": path,
            "text": text,
            "sha256": sha256,
        },
        "environment": {
            "python_version": platform.python_version(),
            "refiner_version": refiner_version,
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        },
        "dependencies": _collect_dependencies(),
    }


__all__ = ["build_run_manifest"]
