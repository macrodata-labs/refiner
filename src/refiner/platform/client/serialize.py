from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cloudpickle

CLOUD_PIPELINE_PAYLOAD_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB


@dataclass(frozen=True, slots=True)
class PreparedPipelinePayload:
    """Serialized pipeline payload prepared for cloud file staging."""

    payload_bytes: bytes
    """Raw cloudpickle payload bytes."""

    sha256: str
    """Lowercase hex SHA-256 digest of payload_bytes."""

    size_bytes: int
    """Exact payload size in bytes."""

    @classmethod
    def from_pipeline(cls, pipeline: object) -> PreparedPipelinePayload:
        # This payload is executable-by-design and must only be deserialized
        # in a trusted tenant boundary for the submitting account.
        payload_bytes = cloudpickle.dumps(pipeline)
        size_bytes = len(payload_bytes)
        if size_bytes > CLOUD_PIPELINE_PAYLOAD_MAX_BYTES:
            raise ValueError(
                "Pipeline payload exceeds cloud upload limit "
                f"({size_bytes} bytes > {CLOUD_PIPELINE_PAYLOAD_MAX_BYTES} bytes)."
            )
        return cls(
            payload_bytes=payload_bytes,
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
            size_bytes=size_bytes,
        )
