from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from refiner.media.types import MediaFile


@dataclass(frozen=True, slots=True)
class Video:
    media: "MediaFile | DecodedVideo"
    from_timestamp_s: float
    to_timestamp_s: float

    @property
    def uri(self) -> str:
        return self.media.uri


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: tuple[np.ndarray, ...]
    uri: str
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = "rgb24"


__all__ = ["Video", "DecodedVideo"]
