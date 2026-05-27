from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    images: bool | None = None
    audio: bool | None = None
    video: bool | None = None
    files: bool | None = None
    tools: bool | None = None
    structured_output: bool | None = None
    reasoning: bool | None = None
    generated_media: bool | None = None
    citations: bool | None = None


__all__ = ["ModelCapabilities"]
