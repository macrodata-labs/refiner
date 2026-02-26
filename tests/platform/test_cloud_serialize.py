from __future__ import annotations

import base64
import hashlib

from refiner.platform.cloud.serialize import serialize_pipeline_inline


def test_serialize_pipeline_inline_returns_base64_and_sha() -> None:
    payload = serialize_pipeline_inline({"hello": "world"})

    raw = base64.b64decode(payload.bytes_b64.encode("ascii"))
    assert payload.format == "cloudpickle"
    assert payload.size_bytes == len(raw)
    assert payload.sha256 == hashlib.sha256(raw).hexdigest()


def test_serialize_pipeline_inline_enforces_size_limit() -> None:
    try:
        serialize_pipeline_inline("x" * 1024, max_bytes=8)
    except ValueError as err:
        assert "inline cloud submission limit" in str(err)
    else:  # pragma: no cover
        raise AssertionError("expected size limit error")
