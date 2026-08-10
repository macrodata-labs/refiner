from __future__ import annotations

import pyarrow as pa

from refiner.pipeline.data.block import Block
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular


def block_to_table(block: Block) -> pa.Table:
    table = (
        block.table
        if isinstance(block, Tabular)
        else (
            Tabular.from_rows(block).table
            if not block
            else block[0].tabular_type.from_rows(block).table
        )
    )
    if SHARD_ID_COLUMN in table.schema.names:
        table = table.drop_columns([SHARD_ID_COLUMN])
    return table
