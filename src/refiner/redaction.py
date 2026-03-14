from __future__ import annotations

from collections.abc import Sequence
from typing import Any

REDACTION_PLACEHOLDER = "REDACTED_KEY"


def redact_captured_text(text: str, *, secret_values: Sequence[str]) -> str:
    redacted = text
    ordered_secrets: list[str] = [value for value in secret_values if value]
    ordered_secrets.sort(key=len, reverse=True)
    for secret_value in ordered_secrets:
        redacted = redacted.replace(secret_value, REDACTION_PLACEHOLDER)
    return redacted


def redact_captured_strings(value: Any, *, secret_values: Sequence[str]) -> Any:
    if not secret_values:
        return value
    if isinstance(value, str):
        return redact_captured_text(value, secret_values=secret_values)
    if isinstance(value, list):
        return [
            redact_captured_strings(item, secret_values=secret_values) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            redact_captured_strings(item, secret_values=secret_values) for item in value
        )
    if isinstance(value, dict):
        return {
            key: redact_captured_strings(item, secret_values=secret_values)
            for key, item in value.items()
        }
    return value


__all__ = ["REDACTION_PLACEHOLDER", "redact_captured_strings", "redact_captured_text"]
