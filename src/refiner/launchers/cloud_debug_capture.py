from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from refiner.launchers.cloud import CloudLauncher, CloudLaunchResult


_ACTIVE_CAPTURE: ContextVar[CloudLaunchCapture | None] = ContextVar(
    "refiner_cloud_debug_capture",
    default=None,
)


@dataclass(slots=True)
class CloudLaunchCapture:
    launchers: list[CloudLauncher] = field(default_factory=list)

    def capture(self, launcher: CloudLauncher) -> CloudLaunchResult:
        from refiner.launchers.cloud import CloudLaunchResult

        self.launchers.append(launcher)
        return CloudLaunchResult(
            job_id="debug-capture",
            stage_index=0,
            status="captured",
            warnings=[],
        )

    def single(self) -> CloudLauncher:
        if not self.launchers:
            raise ValueError(
                "pipeline script must call launch_cloud(...) exactly once; found none"
            )
        if len(self.launchers) != 1:
            raise ValueError(
                "pipeline script must call launch_cloud(...) exactly once; "
                f"found {len(self.launchers)}"
            )
        return self.launchers[0]


@contextmanager
def capture_cloud_launches() -> Iterator[CloudLaunchCapture]:
    capture = CloudLaunchCapture()
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def active_cloud_launch_capture() -> CloudLaunchCapture | None:
    return _ACTIVE_CAPTURE.get()


__all__ = [
    "CloudLaunchCapture",
    "active_cloud_launch_capture",
    "capture_cloud_launches",
]
