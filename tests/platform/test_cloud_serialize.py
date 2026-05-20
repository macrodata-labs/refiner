from __future__ import annotations

import hashlib

import cloudpickle

from refiner.platform.client.serialize import PreparedPipelinePayload


def test_prepared_pipeline_payload_from_pipeline_returns_raw_bytes_and_sha() -> None:
    payload = PreparedPipelinePayload.from_pipeline({"hello": "world"})

    assert cloudpickle.loads(payload.payload_bytes) == {"hello": "world"}
    assert payload.size_bytes == len(payload.payload_bytes)
    assert payload.sha256 == hashlib.sha256(payload.payload_bytes).hexdigest()
