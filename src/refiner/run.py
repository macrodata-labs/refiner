from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RunHandle:
    job_id: str
    stage_index: int
    client: Any | None = None
    workspace_slug: str | None = None
    worker_name: str | None = None
    worker_id: str | None = None

    def with_worker(
        self,
        *,
        worker_name: str | None = None,
        worker_id: str | None = None,
    ) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=self.stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=worker_name if worker_name is not None else self.worker_name,
            worker_id=worker_id if worker_id is not None else self.worker_id,
        )

    def with_stage(self, stage_index: int) -> RunHandle:
        return RunHandle(
            job_id=self.job_id,
            stage_index=stage_index,
            client=self.client,
            workspace_slug=self.workspace_slug,
            worker_name=self.worker_name,
            worker_id=self.worker_id,
        )


__all__ = ["RunHandle"]
