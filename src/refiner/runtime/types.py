from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import pyarrow as pa

if TYPE_CHECKING:
    from refiner.sources.row import Row

TabularBlock: TypeAlias = pa.Table | pa.RecordBatch
SourceUnit: TypeAlias = "Row | TabularBlock"

__all__ = [
    "TabularBlock",
    "SourceUnit",
]
