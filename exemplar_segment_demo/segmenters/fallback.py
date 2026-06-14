from __future__ import annotations

import cv2
import numpy as np

from exemplar_segment_demo.segmenters.base import Segmenter
from exemplar_segment_demo.types import CandidateMask


class FallbackSegmenter(Segmenter):
    """Classical CV proposal generator used when SAM 3 is not available.

    This is intentionally conservative: it produces candidate object regions, not final
    decisions. The matcher later decides which proposals look like the references.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.65,
        max_candidates: int = 80,
        grid_proposals: bool = True,
        grid_sizes: list[int] | None = None,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_candidates = max_candidates
        self.grid_proposals = grid_proposals
        self.grid_sizes = grid_sizes or [2, 3, 4]

    def segment(self, image: np.ndarray) -> list[CandidateMask]:
        h, w = image.shape[:2]
        min_area = max(16, int(h * w * self.min_area_ratio))
        max_area = int(h * w * self.max_area_ratio)

        candidates: list[CandidateMask] = []
        candidates.extend(self._connected_component_proposals(image, min_area, max_area))
        candidates.extend(self._grabcut_center_proposal(image, min_area, max_area))
        if self.grid_proposals:
            candidates.extend(self._grid_proposals(h, w, min_area, max_area))

        candidates = self._deduplicate(candidates)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self.max_candidates]

    def _connected_component_proposals(
        self, image: np.ndarray, min_area: int, max_area: int
    ) -> list[CandidateMask]:
        h, w = image.shape[:2]
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

        masks: list[np.ndarray] = []
        edges = cv2.Canny(gray, 70, 160)
        edge_closed = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )
        masks.append(edge_closed > 0)

        for channel in cv2.split(lab):
            _, otsu = cv2.threshold(
                channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            masks.append(otsu > 0)
            masks.append(otsu == 0)

        proposals: list[CandidateMask] = []
        kernel = np.ones((5, 5), np.uint8)
        for idx, raw_mask in enumerate(masks):
            clean = cv2.morphologyEx(
                raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1
            )
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
            for label in range(1, num_labels):
                x, y, bw, bh, area = stats[label]
                if area < min_area or area > max_area:
                    continue
                if bw < 8 or bh < 8:
                    continue
                mask = labels == label
                compactness = float(area) / float(max(1, bw * bh))
                size_score = min(1.0, area / max(min_area, 1))
                score = 0.45 + 0.35 * compactness + 0.20 * min(size_score, 1.0)
                proposals.append(
                    CandidateMask(
                        mask=mask,
                        bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                        score=float(score),
                        source=f"cc_{idx}",
                    )
                )

        return proposals

    def _grabcut_center_proposal(
        self, image: np.ndarray, min_area: int, max_area: int
    ) -> list[CandidateMask]:
        h, w = image.shape[:2]
        if h < 32 or w < 32:
            return []

        rect = (
            int(w * 0.08),
            int(h * 0.08),
            max(1, int(w * 0.84)),
            max(1, int(h * 0.84)),
        )
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                mask,
                rect,
                bgd_model,
                fgd_model,
                3,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            return []

        fg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
        area = int(fg.sum())
        if area < min_area or area > max_area:
            return []

        bbox = _bbox_from_mask(fg)
        if bbox is None:
            return []
        return [CandidateMask(mask=fg, bbox=bbox, score=0.55, source="grabcut_center")]

    def _grid_proposals(
        self, h: int, w: int, min_area: int, max_area: int
    ) -> list[CandidateMask]:
        proposals: list[CandidateMask] = []
        for grid_size in self.grid_sizes:
            cell_h = h / grid_size
            cell_w = w / grid_size
            for gy in range(grid_size):
                for gx in range(grid_size):
                    x1 = int(gx * cell_w)
                    y1 = int(gy * cell_h)
                    x2 = int((gx + 1) * cell_w)
                    y2 = int((gy + 1) * cell_h)
                    area = (x2 - x1) * (y2 - y1)
                    if area < min_area or area > max_area:
                        continue
                    mask = np.zeros((h, w), dtype=bool)
                    mask[y1:y2, x1:x2] = True
                    proposals.append(
                        CandidateMask(
                            mask=mask,
                            bbox=(x1, y1, x2, y2),
                            score=0.25,
                            source=f"grid_{grid_size}",
                        )
                    )
        return proposals

    def _deduplicate(self, candidates: list[CandidateMask]) -> list[CandidateMask]:
        kept: list[CandidateMask] = []
        for candidate in candidates:
            duplicate = False
            for existing in kept:
                if _bbox_iou(candidate.bbox, existing.bbox) > 0.86:
                    duplicate = True
                    if candidate.score > existing.score:
                        existing.mask = candidate.mask
                        existing.bbox = candidate.bbox
                        existing.score = candidate.score
                        existing.source = candidate.source
                    break
            if not duplicate:
                kept.append(candidate)
        return kept


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _bbox_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union == 0 else inter / union
