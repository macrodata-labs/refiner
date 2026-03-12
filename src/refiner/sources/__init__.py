from __future__ import annotations

from .base import BaseSource
from .items import ItemsSource
from .row import ArrowRowView, DictRow, Row
from .task import TaskSource

__all__ = [
    "BaseSource",
    "Row",
    "DictRow",
    "ArrowRowView",
    "BaseReader",
    "CsvReader",
    "JsonlReader",
    "LeRobotEpisodeReader",
    "ParquetReader",
    "ItemsSource",
    "TaskSource",
]


def __getattr__(name: str):
    if name == "BaseReader":
        from .readers.base import BaseReader

        return BaseReader
    if name == "CsvReader":
        from .readers.csv import CsvReader

        return CsvReader
    if name == "JsonlReader":
        from .readers.jsonl import JsonlReader

        return JsonlReader
    if name == "LeRobotEpisodeReader":
        from .readers.lerobot import LeRobotEpisodeReader

        return LeRobotEpisodeReader
    if name == "ParquetReader":
        from .readers.parquet import ParquetReader

        return ParquetReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
