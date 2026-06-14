from __future__ import annotations

import numpy as np

from exemplar_segment_demo.segmenters.base import Segmenter
from exemplar_segment_demo.types import CandidateMask


class SAM3Segmenter(Segmenter):
    """Placeholder adapter for a future SAM 3 implementation.

    Keep the public contract identical to FallbackSegmenter: RGB image in,
    CandidateMask list out. Once local SAM 3 weights/API are available, put the
    model loading in __init__ and translate model outputs in segment().
    """

    def __init__(self, **_: object) -> None:
        raise RuntimeError(
            "SAM 3 segmenter is not wired in this demo yet. "
            "Set segmenter.name=fallback in configs/demo.yaml or implement this adapter."
        )

    def segment(self, image: np.ndarray) -> list[CandidateMask]:
        raise NotImplementedError
