from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from exemplar_segment_demo.types import CandidateMask, MatchResult


AggregationMode = Literal["max", "topk_average"]


@dataclass
class ExemplarMatcher:
    embedding_model: str = "color_hist"
    histogram_bins: tuple[int, int, int] = (8, 8, 8)
    score_aggregation: AggregationMode = "topk_average"
    top_k: int = 3
    threshold: float = 0.72
    nms_iou_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.embedding_model != "color_hist":
            raise ValueError(
                f"Unsupported embedding_model={self.embedding_model!r}. "
                "This minimal demo ships with color_hist. Replace ExemplarMatcher "
                "or add a DINOv2-backed embedder here."
            )

    def embed_references(self, reference_images: list[np.ndarray]) -> np.ndarray:
        if not reference_images:
            raise ValueError("No reference images were provided.")
        embeddings = [self.embed_crop(image) for image in reference_images]
        return np.stack(embeddings, axis=0)

    def match_candidates(
        self,
        image: np.ndarray,
        candidates: list[CandidateMask],
        reference_embeddings: np.ndarray,
    ) -> list[MatchResult]:
        matches: list[MatchResult] = []
        for candidate in candidates:
            crop = crop_candidate(image, candidate)
            if crop.size == 0:
                continue
            embedding = self.embed_crop(crop)
            similarities = reference_embeddings @ embedding
            score = self.aggregate_scores(similarities)
            if score >= self.threshold:
                matches.append(
                    MatchResult(candidate=candidate, similarity=float(score), crop=crop)
                )
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return nms_matches(matches, self.nms_iou_threshold)

    def embed_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        if self.embedding_model == "color_hist":
            return color_hist_embedding(crop_rgb, self.histogram_bins)
        raise AssertionError("unreachable")

    def aggregate_scores(self, similarities: np.ndarray) -> float:
        if similarities.size == 0:
            return 0.0
        if self.score_aggregation == "max":
            return float(np.max(similarities))
        if self.score_aggregation == "topk_average":
            k = max(1, min(int(self.top_k), similarities.size))
            top = np.partition(similarities, -k)[-k:]
            return float(np.mean(top))
        raise ValueError(f"Unknown score_aggregation: {self.score_aggregation}")


def color_hist_embedding(
    crop_rgb: np.ndarray, bins: tuple[int, int, int] = (8, 8, 8)
) -> np.ndarray:
    crop = crop_rgb
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Expected an RGB crop with shape HxWx3.")

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    # Product references and masked crops are often on white backgrounds. Ignoring
    # near-white pixels prevents empty/background crops from looking like references.
    background = np.logical_and(hsv[..., 1] < 30, hsv[..., 2] > 235)
    valid = np.logical_not(background).astype(np.uint8)
    hist_mask = valid if int(valid.sum()) >= 20 else None
    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=hist_mask,
        histSize=list(bins),
        ranges=[0, 180, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).flatten()
    hist_norm = float(np.linalg.norm(hist))
    if hist_norm > 1e-12:
        hist = hist / hist_norm

    # A tiny shape/texture supplement keeps all-white/all-black items from collapsing
    # into nearly identical vectors.
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    texture = resized_gray.astype(np.float32).flatten() / 255.0
    texture = texture - float(texture.mean())
    texture_norm = float(np.linalg.norm(texture))
    if texture_norm > 1e-12:
        texture = texture / texture_norm

    vector = np.concatenate([hist, 0.15 * texture], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def crop_candidate(image: np.ndarray, candidate: CandidateMask, padding: float = 0.05) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = candidate.bbox
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return crop

    mask_crop = candidate.mask[y1:y2, x1:x2]
    if mask_crop.shape[:2] != crop.shape[:2]:
        return crop

    background = np.full_like(crop, 255)
    return np.where(mask_crop[..., None], crop, background)


def nms_matches(matches: list[MatchResult], iou_threshold: float) -> list[MatchResult]:
    if iou_threshold <= 0:
        return matches
    kept: list[MatchResult] = []
    for match in matches:
        if all(_bbox_iou(match.candidate.bbox, item.candidate.bbox) <= iou_threshold for item in kept):
            kept.append(match)
    return kept


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
