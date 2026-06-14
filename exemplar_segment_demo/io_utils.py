from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path: Path, recursive: bool = True) -> list[Path]:
    if not path.exists():
        return []
    paths = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        p
        for p in paths
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not is_hidden_path(p, root=path)
    )


def is_hidden_path(path: Path, root: Path | None = None) -> bool:
    try:
        parts = path.relative_to(root).parts if root else path.parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def read_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_png_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask_u8)


def write_image_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")
