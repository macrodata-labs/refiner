from __future__ import annotations

import base64
import hashlib

import cloudpickle

from .models import CloudPipelinePayload

INLINE_PIPELINE_PAYLOAD_MAX_BYTES = 1_000_000


def serialize_pipeline_inline(
    pipeline: object,
    *,
    max_bytes: int = INLINE_PIPELINE_PAYLOAD_MAX_BYTES,
) -> CloudPipelinePayload:
    # This payload is executable-by-design and must only be deserialized
    # in a trusted tenant boundary for the submitting account.
    payload_bytes = cloudpickle.dumps(pipeline)
    size_bytes = len(payload_bytes)
    if size_bytes > max_bytes:
        raise ValueError(
            "Pipeline payload exceeds inline cloud submission limit "
            f"({size_bytes} bytes > {max_bytes} bytes). "
            "Artifact uploads are not implemented yet."
        )

    return CloudPipelinePayload(
        format="cloudpickle",
        bytes_b64=base64.b64encode(payload_bytes).decode("ascii"),
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        size_bytes=size_bytes,
    )
