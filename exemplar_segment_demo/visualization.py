from __future__ import annotations

import cv2
import numpy as np

from exemplar_segment_demo.types import MatchResult


PALETTE = np.array(
    [
        [230, 57, 70],
        [42, 157, 143],
        [69, 123, 157],
        [244, 162, 97],
        [131, 56, 236],
        [255, 183, 3],
    ],
    dtype=np.uint8,
)


def draw_matches(
    image: np.ndarray, matches: list[MatchResult], alpha: float = 0.45
) -> np.ndarray:
    vis = image.copy()
    overlay = image.copy()

    for idx, match in enumerate(matches):
        color = PALETTE[idx % len(PALETTE)]
        mask = match.candidate.mask.astype(bool)
        overlay[mask] = color

    vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)

    for idx, match in enumerate(matches):
        x1, y1, x2, y2 = match.candidate.bbox
        color = tuple(int(c) for c in PALETTE[idx % len(PALETTE)].tolist())
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{idx}: {match.similarity:.3f}"
        text_origin = (x1, max(16, y1 - 6))
        cv2.putText(
            vis,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def combined_mask(matches: list[MatchResult], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for match in matches:
        mask |= match.candidate.mask.astype(bool)
    return mask
