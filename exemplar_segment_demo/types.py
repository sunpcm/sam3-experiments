from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateMask:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    source: str = "unknown"


@dataclass
class MatchResult:
    candidate: CandidateMask
    similarity: float
    crop: np.ndarray
