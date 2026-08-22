from __future__ import annotations

from typing import Final

_STEP_INDEX_ATTR: Final = "_refiner_step_index"
_MAX_LIFECYCLE_ERROR_CHARS: Final = 512


def annotate_step_error(error: BaseException, step_index: int | None) -> None:
    if step_index is None:
        return
    try:
        if getattr(error, _STEP_INDEX_ATTR, None) is None:
            setattr(error, _STEP_INDEX_ATTR, step_index)
    except Exception:
        pass


def describe_lifecycle_error(error: BaseException) -> str:
    try:
        message = str(error).strip()
    except Exception:
        message = ""
    message = message or type(error).__name__
    if len(message) > _MAX_LIFECYCLE_ERROR_CHARS:
        message = f"{message[: _MAX_LIFECYCLE_ERROR_CHARS - 1]}…"

    try:
        step_index = getattr(error, _STEP_INDEX_ATTR, None)
    except Exception:
        step_index = None
    if isinstance(step_index, int):
        return f"{message} | step_index={step_index}"
    return message


__all__ = ["annotate_step_error", "describe_lifecycle_error"]
