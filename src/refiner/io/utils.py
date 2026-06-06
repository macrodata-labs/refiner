from __future__ import annotations

from fsspec import AbstractFileSystem


_PROTOCOL_REFINER_EXTRAS = {
    "s3": "s3",
    "s3a": "s3",
    "hf": "hf",
    "gcs": "gcs",
    "gs": "gcs",
}


def required_refiner_extras(path: str, fs: AbstractFileSystem) -> tuple[str, ...]:
    protocol = fs.protocol
    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
    path_protocol, sep, _rest = path.partition("://")
    return tuple(
        sorted(
            {
                extra
                for item in (*protocols, path_protocol if sep else None)
                if item is not None
                and (extra := _PROTOCOL_REFINER_EXTRAS.get(str(item).lower()))
                is not None
            }
        )
    )
