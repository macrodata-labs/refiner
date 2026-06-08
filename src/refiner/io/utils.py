from __future__ import annotations

from fsspec import AbstractFileSystem
from fsspec.core import split_protocol


_PROTOCOL_REFINER_EXTRAS = {
    "s3": "s3",
    "s3a": "s3",
    "hf": "hf",
    "gcs": "gcs",
    "gs": "gcs",
}


def required_refiner_extras(
    path: str,
    fs: AbstractFileSystem | None = None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    path_bits = path.split("::")
    for bit in path_bits:
        protocol, _path = split_protocol(bit)
        if protocol is not None:
            candidates.append(protocol)
        elif len(path_bits) > 1 and bit in _PROTOCOL_REFINER_EXTRAS:
            candidates.append(bit)
    if fs is not None:
        seen: set[int] = set()
        filesystems = [fs]
        while filesystems:
            current = filesystems.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            protocol = current.protocol
            candidates.extend((protocol,) if isinstance(protocol, str) else protocol)
            target_protocol = getattr(current, "target_protocol", None)
            if isinstance(target_protocol, str):
                candidates.append(target_protocol)
            elif target_protocol is not None:
                candidates.extend(target_protocol)
            target_fs = getattr(current, "fs", None)
            if isinstance(target_fs, AbstractFileSystem):
                filesystems.append(target_fs)
    return tuple(
        sorted(
            {
                extra
                for protocol in candidates
                if (extra := _PROTOCOL_REFINER_EXTRAS.get(protocol.lower())) is not None
            }
        )
    )
