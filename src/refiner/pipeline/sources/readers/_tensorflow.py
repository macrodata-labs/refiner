from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from refiner.utils import check_required_dependencies


def require_tensorflow():
    check_required_dependencies(
        "read_tfrecords",
        [("tensorflow", "tensorflow")],
        dist="tensorflow",
    )
    return importlib.import_module("tensorflow")


def require_tfds():
    check_required_dependencies(
        "read_tfds",
        [
            ("tensorflow", "tensorflow"),
            ("tensorflow_datasets", "tensorflow-datasets"),
        ],
        dist="tfds",
    )
    return (
        importlib.import_module("tensorflow"),
        importlib.import_module("tensorflow_datasets"),
    )


def tensorflow_batch_to_table(batch: Mapping[str, Any]) -> pa.Table:
    tf = importlib.import_module("tensorflow")
    return pa.table({name: _column_values(value, tf) for name, value in batch.items()})


def _column_values(value: Any, tf) -> pa.Array | list[Any]:
    if isinstance(value, Mapping):
        names = []
        columns = []
        for name, child in value.items():
            column = _column_values(child, tf)
            names.append(name)
            columns.append(column if isinstance(column, pa.Array) else pa.array(column))
        if not columns:
            return []
        return pa.StructArray.from_arrays(columns, names=names)

    if isinstance(value, tf.SparseTensor):
        return tf.RaggedTensor.from_sparse(value).to_list()
    if isinstance(value, tf.RaggedTensor):
        return value.to_list()
    array = value.numpy()
    if array.ndim == 1:
        return pa.array(array)
    if array.dtype != object:
        array_type = pa.from_numpy_dtype(array.dtype)
        for dim in reversed(array.shape[1:]):
            array_type = pa.list_(array_type, int(dim))
        return pa.array(array.tolist(), type=array_type)
    return array.tolist()


__all__ = [
    "require_tensorflow",
    "require_tfds",
    "tensorflow_batch_to_table",
]
