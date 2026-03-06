from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import subprocess
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RefinerRuntimeMetadata:
    version: str | None = None
    git_sha: str | None = None


def _resolve_installed_version() -> str | None:
    for package_name in ("refiner", "macrodata-refiner"):
        try:
            version = metadata.version(package_name).strip()
        except metadata.PackageNotFoundError:
            continue
        if version:
            return version
    return None


def _resolve_direct_url_git_sha() -> str | None:
    for package_name in ("refiner", "macrodata-refiner"):
        try:
            dist = metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            continue
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        vcs_info = data.get("vcs_info")
        if not isinstance(vcs_info, dict):
            continue
        commit = str(vcs_info.get("commit_id", "")).strip()
        if commit:
            return commit
    return None


def _resolve_repo_root(start: Path) -> Path | None:
    for parent in start.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_local_repo_git_sha() -> str | None:
    repo_root = _resolve_repo_root(Path(__file__))
    if repo_root is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


@lru_cache(maxsize=1)
def resolve_refiner_runtime_metadata() -> RefinerRuntimeMetadata:
    return RefinerRuntimeMetadata(
        version=_resolve_installed_version(),
        git_sha=_resolve_direct_url_git_sha() or _resolve_local_repo_git_sha(),
    )
