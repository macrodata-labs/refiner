from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4
import re
import time

if TYPE_CHECKING:
    from refiner.ledger import BaseLedger
    from refiner.ledger.shard import Shard
    from refiner.pipeline import RefinerPipeline


class BaseLauncher(ABC):
    def __init__(
        self, *, pipeline: RefinerPipeline, name: str, run_id: str | None = None
    ):
        if not name.strip():
            raise ValueError("name must be non-empty")
        self.pipeline = pipeline
        self.name = name
        self.run_id = run_id or self._build_run_id(name)
        self.ledger: BaseLedger | None = None

    @staticmethod
    def _build_run_id(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-") or "job"
        return f"{slug}-{int(time.time())}-{uuid4().hex[:8]}"

    def seed_ledger(self, *, shards: list["Shard"]) -> None:
        if self.ledger is None:
            raise ValueError("launcher.ledger is not configured")
        self.ledger.seed_shards(shards)

    @abstractmethod
    def launch(self):
        raise NotImplementedError


__all__ = ["BaseLauncher"]
