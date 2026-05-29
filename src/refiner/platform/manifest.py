from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request
from collections.abc import Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

_REDACTION_PLACEHOLDER = "REDACTED_SECRET"
_NORMALIZED_DEPENDENCY_SEPARATOR_PATTERN = re.compile(r"[-_.]+")


def _redact_captured_text(text: str, *, secret_values: Sequence[str]) -> str:
    redacted = text
    # Replace longer secrets first so substring overlaps do not leak suffixes.
    for secret_value in sorted(
        secret_values, key=lambda value: len(value), reverse=True
    ):
        if secret_value != "":
            redacted = redacted.replace(secret_value, _REDACTION_PLACEHOLDER)
    return redacted


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


def _dependency_key(name: str) -> str:
    try:
        package_name = Requirement(name).name
    except InvalidRequirement:
        package_name = name.split("[", 1)[0].strip()
    return _NORMALIZED_DEPENDENCY_SEPARATOR_PATTERN.sub("-", package_name).lower()


def _merge_extra_dependencies(
    dependencies: list[dict[str, str]],
    extra_dependencies: Sequence[str] | None,
) -> list[dict[str, str]]:
    if not extra_dependencies:
        return dependencies
    if isinstance(extra_dependencies, str):
        raise ValueError("extra_dependencies must be a sequence of requirement strings")

    merged = {_dependency_key(dep["name"]): dict(dep) for dep in dependencies}
    for dependency in extra_dependencies:
        text = str(dependency).strip()
        if not text:
            raise ValueError("extra_dependencies contains an empty dependency name")
        try:
            requirement = Requirement(text)
        except InvalidRequirement as err:
            raise ValueError(
                f"extra_dependencies contains invalid requirement {text!r}"
            ) from err

        specifiers = list(requirement.specifier)
        if len(specifiers) == 1 and specifiers[0].operator == "==":
            name = requirement.name
            if requirement.extras:
                name = f"{name}[{','.join(sorted(requirement.extras))}]"
            merged[_dependency_key(requirement.name)] = {
                "name": name,
                "version": specifiers[0].version,
            }
        else:
            merged[_dependency_key(requirement.name)] = {"name": text}
    return list(merged.values())


def _resolve_installed_version() -> str | None:
    try:
        version = importlib_metadata.version("macrodata-refiner").strip()
    except importlib_metadata.PackageNotFoundError:
        return None
    return version or None


def _resolve_direct_url_git_sha() -> str | None:
    try:
        dist = importlib_metadata.distribution("macrodata-refiner")
    except importlib_metadata.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    vcs_info = data.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit = str(vcs_info.get("commit_id", "")).strip()
    return commit or None


def _resolve_repo_root(start: Path) -> Path | None:
    for parent in start.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_local_repo_git_sha() -> str | None:
    repo_root = _resolve_repo_root(Path(__file__))
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def refiner_ref_exists_on_remote(ref: str) -> bool:
    request = urllib_request.Request(
        f"https://api.github.com/repos/macrodata-labs/refiner/commits/{ref}"
    )
    try:
        with urllib_request.urlopen(request):
            return True
    except (urllib_error.HTTPError, urllib_error.URLError):
        return False


def build_run_manifest(
    *,
    secret_values: Sequence[str] = (),
    capture_dependencies: bool = True,
    extra_dependencies: Sequence[str] | None = None,
) -> dict[str, Any]:
    script_path = _detect_script_path()
    path, text, sha256 = _read_script(script_path)
    refiner_version = _resolve_installed_version()
    refiner_ref = _resolve_direct_url_git_sha() or _resolve_local_repo_git_sha()

    manifest: dict[str, Any] = {
        "version": 1,
        "script": {
            "path": path,
            "text": _redact_captured_text(text, secret_values=secret_values)
            if isinstance(text, str) and secret_values
            else text,
            "sha256": sha256,
        },
        "environment": {
            "python_version": platform.python_version(),
            "refiner_version": refiner_version,
            "refiner_ref": refiner_ref,
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        },
    }
    dependencies = _collect_dependencies() if capture_dependencies else []
    if capture_dependencies or extra_dependencies:
        manifest["dependencies"] = _merge_extra_dependencies(
            dependencies,
            extra_dependencies,
        )
    return manifest


__all__ = [
    "build_run_manifest",
    "_redact_captured_text",
]
