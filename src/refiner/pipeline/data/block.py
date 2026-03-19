from __future__ import annotations

from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular

Block = list[Row] | Tabular
StreamItem = Row | Block

__all__ = ["Block", "StreamItem"]
