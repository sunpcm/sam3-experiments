from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from exemplar_segment_demo.types import CandidateMask


class Segmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> list[CandidateMask]:
        """Return candidate instance masks for an RGB image."""
