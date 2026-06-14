from __future__ import annotations

from typing import Any

from exemplar_segment_demo.segmenters.base import Segmenter
from exemplar_segment_demo.segmenters.fallback import FallbackSegmenter
from exemplar_segment_demo.segmenters.sam3 import SAM3Segmenter


def build_segmenter(config: dict[str, Any]) -> Segmenter:
    name = str(config.get("name", "fallback")).lower()
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name == "fallback":
        return FallbackSegmenter(**kwargs)
    if name in {"sam3", "sam_3"}:
        return SAM3Segmenter(**kwargs)
    raise ValueError(f"Unknown segmenter: {name}")
