from __future__ import annotations

from dataclasses import dataclass

from refiner.worker.errors import annotate_step_error, describe_lifecycle_error


@dataclass(frozen=True)
class _FrozenError(Exception):
    message: str


def test_annotate_step_error_does_not_replace_immutable_error() -> None:
    error = _FrozenError("original failure")

    annotate_step_error(error, 2)

    assert describe_lifecycle_error(error) == "original failure"


def test_describe_lifecycle_error_caps_message_and_includes_step() -> None:
    error = RuntimeError("x" * 600)
    annotate_step_error(error, 2)

    description = describe_lifecycle_error(error)

    assert description == f"{'x' * 511}… | step_index=2"


__all__: list[str] = []
