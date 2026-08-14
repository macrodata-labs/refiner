from __future__ import annotations

from urllib.parse import urlsplit

import pyarrow as pa

from refiner.pipeline.data.block import Block, strip_internal_columns
from refiner.pipeline.data.tabular import Tabular


def validate_lance_uri(uri: str) -> None:
    parsed = urlsplit(uri)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Lance URIs must not contain credentials, query parameters, or fragments; "
            "configure storage credentials through the environment or Lance settings"
        )


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
    return strip_internal_columns(table)
