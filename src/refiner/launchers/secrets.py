from __future__ import annotations

import ast
import builtins
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


SecretMapping = Mapping[str, object | None]
SecretPayload = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Secrets:
    """Cloud secret source for Macrodata jobs."""

    _kind: Literal["dict", "env"]
    _values: builtins.dict[str, object | None] | None = None
    _env_name: str | None = None
    _keys: tuple[str, ...] | None = None

    @classmethod
    def dict(cls, values: SecretMapping) -> "Secrets":
        return cls(_kind="dict", _values=dict(values))

    @classmethod
    def dotenv(cls, path: str | os.PathLike[str]) -> "Secrets":
        return cls.dict(_load_dotenv(Path(path)))

    @classmethod
    def env(cls, name: str = "default", keys: Sequence[str] | None = None) -> "Secrets":
        if not name:
            raise ValueError("secret environment name is required")
        if isinstance(keys, str):
            raise TypeError("secret environment keys must be a sequence of strings")
        return cls(
            _kind="env", _env_name=name, _keys=tuple(keys) if keys is not None else None
        )


SecretInput = Secrets | SecretMapping | Sequence[Secrets | SecretMapping]


def normalize_secret_sources(secrets: SecretInput | None) -> tuple[Secrets, ...]:
    if secrets is None:
        return ()
    if isinstance(secrets, Secrets):
        return (secrets,)
    if isinstance(secrets, Mapping):
        return (Secrets.dict(cast(SecretMapping, secrets)),)
    normalized: list[Secrets] = []
    for source in secrets:
        if isinstance(source, Secrets):
            normalized.append(source)
        elif isinstance(source, Mapping):
            normalized.append(Secrets.dict(source))
        else:
            raise TypeError("secrets entries must be Secrets or mappings")
    return tuple(normalized)


def resolve_secret_sources(
    sources: Sequence[Secrets],
) -> tuple[list[SecretPayload] | None, tuple[str, ...]]:
    payloads: list[SecretPayload] = []
    redaction_values: list[str] = []
    for source in sources:
        if source._kind == "dict":
            resolved = resolve_env_mapping(source._values or {})
            if resolved:
                payloads.append(resolved)
                redaction_values.extend(resolved.values())
            continue
        payload: SecretPayload = {
            "__type__": "__envkeys__",
            "envname": source._env_name or "default",
        }
        if source._keys is not None:
            payload["keys"] = list(source._keys)
        payloads.append(payload)
    return payloads or None, tuple(redaction_values)


def resolve_env_mapping(values: SecretMapping) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name, value in values.items():
        resolved_value = os.environ.get(name) if value is None else str(value)
        if resolved_value is None:
            raise ValueError(
                f"cloud env {name!r} was set to None but is not present in the environment. Make sure it is being exported."
            )
        resolved[name] = resolved_value
    return resolved


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_dotenv_value(raw_value)
    return values


def _parse_dotenv_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip(value[0])
        return str(parsed)

    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


__all__ = ["Secrets"]
