from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TextIO

_ATTACH_MODE_ENV_VAR = "REFINER_ATTACH"
_VALID_ATTACH_MODES = frozenset({"auto", "attach", "detach"})


class CloudAttachDetached(KeyboardInterrupt):
    pass


@dataclass(frozen=True, slots=True)
class CloudAttachContext:
    job_id: str
    job_name: str
    tracking_url: str
    stage_index: int | None


def normalize_attach_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _VALID_ATTACH_MODES:
        allowed = ", ".join(sorted(_VALID_ATTACH_MODES))
        raise ValueError(
            f"unsupported attach mode {mode!r}; expected one of: {allowed}"
        )
    return normalized


def attach_mode_override() -> str | None:
    value = os.environ.get(_ATTACH_MODE_ENV_VAR)
    if value is None:
        return None
    return normalize_attach_mode(value)


def resolve_launcher_attach_mode(*, interactive: bool) -> str:
    override = attach_mode_override()
    if override is None:
        return "detach"
    if override == "auto":
        return "attach" if interactive else "detach"
    return override


def emit_cloud_followup_commands(
    *,
    context: CloudAttachContext,
    file: TextIO | None = None,
) -> None:
    output = sys.stdout if file is None else file
    print("Cloud job submitted.", file=output)
    print(f"Job ID: {context.job_id}", file=output)
    print(f"URL: {context.tracking_url}", file=output)
    print(f"Attach: macrodata jobs attach {context.job_id}", file=output)
    print(f"Summary: macrodata jobs get {context.job_id}", file=output)
    if context.stage_index is None:
        print(f"Logs: macrodata jobs logs {context.job_id}", file=output)
    else:
        print(
            f"Logs: macrodata jobs logs {context.job_id} --stage {context.stage_index}",
            file=output,
        )
    print(f"Workers: macrodata jobs workers {context.job_id}", file=output)
    print(f"Cancel: macrodata jobs cancel {context.job_id}", file=output)
