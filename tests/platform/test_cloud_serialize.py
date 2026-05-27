from __future__ import annotations

import hashlib

import cloudpickle
import pytest

from refiner.platform.client.serialize import PreparedPipelinePayload


def test_prepared_pipeline_payload_from_pipeline_returns_raw_bytes_and_sha() -> None:
    payload = PreparedPipelinePayload.from_pipeline({"hello": "world"})

    assert cloudpickle.loads(payload.payload_bytes) == {"hello": "world"}
    assert payload.size_bytes == len(payload.payload_bytes)
    assert payload.sha256 == hashlib.sha256(payload.payload_bytes).hexdigest()


def test_prepared_pipeline_payload_from_pipeline_enforces_size_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "refiner.platform.client.serialize.CLOUD_PIPELINE_PAYLOAD_MAX_BYTES", 8
    )
    monkeypatch.setattr(
        "refiner.platform.client.serialize.cloudpickle.dumps",
        lambda pipeline: b"x" * 9,
    )

    with pytest.raises(ValueError, match="cloud upload limit"):
        PreparedPipelinePayload.from_pipeline({"hello": "world"})
