from __future__ import annotations

import hashlib


def worker_token(worker_id: str) -> str:
    digest = hashlib.blake2b(digest_size=6)
    digest.update(worker_id.encode("utf-8"))
    return digest.hexdigest()


__all__ = ["worker_token"]
