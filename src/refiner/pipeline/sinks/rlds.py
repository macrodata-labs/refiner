from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput


def require_tensorflow(command: str = "write_rlds"):
    check_required_dependencies(
        command, [("tensorflow", "tensorflow")], dist="tensorflow"
    )
    return importlib.import_module("tensorflow")


class RldsSink(BaseSink):
    """Write robotics episodes as RLDS-style TFRecord examples."""

    def __init__(
        self,
        output: DataFolderLike,
        *,
        filename_template: str = "{shard_id}__w{worker_id}.tfrecord",
        compression: str | None = None,
    ):
        self.output = DataFolder.resolve(output)
        self.filename_template = filename_template
        self.compression = compression
        self._writers: dict[str, Any] = {}
        self._tf: Any | None = None

    def _relpath(self, shard_id: str) -> str:
        return self.filename_template.format(
            shard_id=shard_id,
            worker_id=get_active_worker_token(),
        )

    def _writer(self, shard_id: str):
        writer = self._writers.get(shard_id)
        if writer is not None:
            return writer
        tf = self._tensorflow()
        options = (
            None
            if self.compression is None
            else tf.io.TFRecordOptions(compression_type=self.compression.upper())
        )
        writer = tf.io.TFRecordWriter(
            self.output.abs_path(self._relpath(shard_id)),
            options=options,
        )
        self._writers[shard_id] = writer
        return writer

    def _tensorflow(self):
        if self._tf is None:
            self._tf = require_tensorflow("write_rlds")
        return self._tf

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        rows = block if not isinstance(block, Tabular) else list(block)
        writer = self._writer(shard_id)
        count = 0
        for row in rows:
            writer.write(self._example(row).SerializeToString())
            count += 1
        return count

    def _example(self, row: Row):
        if not isinstance(row, RoboticsRow):
            raise ValueError("write_rlds requires RoboticsRow inputs")
        tf = self._tensorflow()
        features: dict[str, Any] = {
            "episode_id": _bytes_feature(tf, row.episode_id),
            "length": _int_feature(tf, [row.num_frames]),
            "steps/is_first": _int_feature(tf, _step_flags(row.num_frames, first=True)),
            "steps/is_last": _int_feature(tf, _step_flags(row.num_frames, last=True)),
            "steps/is_terminal": _int_feature(
                tf, _step_flags(row.num_frames, last=True)
            ),
        }
        if row.task is not None:
            features["task"] = _bytes_feature(tf, row.task)
            features["language_instruction"] = _bytes_feature(tf, row.task)
        if row.fps is not None:
            features["fps"] = _float_feature(tf, [float(row.fps)])
        if row.robot_type is not None:
            features["robot_type"] = _bytes_feature(tf, row.robot_type)
        _add_array_features(tf, features, "steps/action", row.actions)
        _add_array_features(tf, features, "steps/observation/state", row.states)
        _add_array_features(tf, features, "steps/timestamp", row.timestamps)
        return tf.train.Example(features=tf.train.Features(feature=features))

    def on_shard_complete(self, shard_id: str) -> None:
        writer = self._writers.pop(shard_id, None)
        if writer is not None:
            writer.close()
            log_throughput("files_written", 1, shard_id=shard_id, unit="files")

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def describe(self) -> tuple[str, str, dict[str, object]]:
        args: dict[str, object] = {
            "path": self.output.abs_path(),
            "filename_template": self.filename_template,
        }
        if self.compression is not None:
            args["compression"] = self.compression
        return ("write_rlds", "writer", args)

    def build_reducer(self) -> BaseSink | None:
        return FileCleanupReducerSink(
            output=self.output,
            filename_template=self.filename_template,
            reducer_name="write_rlds_reduce",
        )


def _step_flags(length: int, *, first: bool = False, last: bool = False) -> list[int]:
    if length <= 0:
        return []
    values = [0] * length
    if first:
        values[0] = 1
    if last:
        values[-1] = 1
    return values


def _add_array_features(
    tf,
    features: dict[str, Any],
    name: str,
    values: Any,
) -> None:
    if values is None:
        return
    array = _array(values)
    features[name] = _float_feature(tf, array.reshape(-1).astype(float).tolist())
    features[f"{name}/shape"] = _int_feature(tf, list(array.shape))


def _array(values: Any) -> np.ndarray:
    if isinstance(values, pa.ChunkedArray | pa.Array):
        return np.asarray(values.to_pylist())
    return np.asarray(values)


def _bytes_feature(tf, value: str | bytes):
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def _float_feature(tf, values: list[float]):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int_feature(tf, values: list[int]):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


__all__ = ["RldsSink"]
