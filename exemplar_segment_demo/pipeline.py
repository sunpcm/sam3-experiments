from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from exemplar_segment_demo.config import DemoConfig, load_config
from exemplar_segment_demo.io_utils import (
    ensure_dirs,
    is_hidden_path,
    list_images,
    read_image_rgb,
    safe_stem,
    write_image_rgb,
    write_json,
    write_png_mask,
)
from exemplar_segment_demo.matcher import ExemplarMatcher
from exemplar_segment_demo.segmenters import build_segmenter
from exemplar_segment_demo.visualization import combined_mask, draw_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplar-guided product mask demo")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config")
    parser.add_argument(
        "--target-id",
        default=None,
        help="Override data.target_id. Uses data/references/<target-id>/*.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override matcher.threshold",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.threshold is not None:
        config.raw["matcher"]["threshold"] = args.threshold
    if args.target_id is not None:
        config.raw["data"]["target_id"] = args.target_id

    run_demo(config)


def run_demo(config: DemoConfig) -> dict[str, Any]:
    output_dirs = {
        "root": config.outputs_dir,
        "masks": config.outputs_dir / "masks",
        "crops": config.outputs_dir / "crops",
        "visualizations": config.outputs_dir / "visualizations",
    }
    ensure_dirs(output_dirs.values())

    reference_dir = resolve_reference_dir(config.references_dir, config.target_id)
    reference_paths = list_images(reference_dir)
    production_paths = list_images(config.production_dir)

    if not reference_paths:
        raise FileNotFoundError(
            f"No reference images found under {reference_dir}. "
            "Expected images directly under data/references/ or under "
            "data/references/<target_id>/."
        )
    if not production_paths:
        raise FileNotFoundError(
            f"No production images found under {config.production_dir}."
        )

    reference_images = [read_image_rgb(path) for path in reference_paths]
    segmenter = build_segmenter(config.raw.get("segmenter", {}))
    matcher = build_matcher(config.raw.get("matcher", {}))
    reference_embeddings = matcher.embed_references(reference_images)

    results: dict[str, Any] = {
        "target_id": reference_dir.name,
        "config": str(config.config_path),
        "references": [str(path) for path in reference_paths],
        "images": [],
    }

    for image_path in tqdm(production_paths, desc="Processing production images"):
        image = read_image_rgb(image_path)
        candidates = segmenter.segment(image)
        matches = matcher.match_candidates(image, candidates, reference_embeddings)
        image_stem = safe_stem(image_path)

        mask_path = output_dirs["masks"] / f"{image_stem}.png"
        vis_path = output_dirs["visualizations"] / f"{image_stem}.jpg"
        write_png_mask(mask_path, combined_mask(matches, image.shape[:2]))
        write_image_rgb(
            vis_path,
            draw_matches(
                image,
                matches,
                alpha=float(config.raw.get("output", {}).get("mask_alpha", 0.45)),
            ),
        )

        image_record: dict[str, Any] = {
            "image": str(image_path),
            "mask_png": str(mask_path),
            "visualization": str(vis_path),
            "instances": [],
        }

        for idx, match in enumerate(matches):
            crop_path = output_dirs["crops"] / f"{image_stem}_inst{idx:03d}.jpg"
            crop_record: str | None = None
            if config.raw.get("output", {}).get("save_candidate_crops", True):
                write_image_rgb(crop_path, match.crop)
                crop_record = str(crop_path)
            image_record["instances"].append(
                {
                    "instance_id": idx,
                    "bbox_xyxy": [int(v) for v in match.candidate.bbox],
                    "similarity": round(float(match.similarity), 6),
                    "segmenter_score": round(float(match.candidate.score), 6),
                    "segmenter_source": match.candidate.source,
                    "crop": crop_record,
                }
            )

        results["images"].append(image_record)

    results_path = config.outputs_dir / "results.json"
    write_json(results_path, results)
    return results


def resolve_reference_dir(references_dir: Path, target_id: str | None) -> Path:
    if target_id:
        return references_dir / target_id

    if not references_dir.exists():
        return references_dir

    direct_images = list_images(references_dir, recursive=False)
    if direct_images:
        return references_dir

    target_dirs = sorted(
        path
        for path in references_dir.iterdir()
        if path.is_dir() and not is_hidden_path(path, root=references_dir)
    )
    if len(target_dirs) == 1:
        return target_dirs[0]
    if len(target_dirs) > 1:
        names = ", ".join(path.name for path in target_dirs)
        raise ValueError(
            "Multiple target_id folders found under references. "
            f"Set data.target_id in config or pass --target-id. Found: {names}"
        )
    return references_dir


def build_matcher(config: dict[str, Any]) -> ExemplarMatcher:
    bins = config.get("histogram_bins", [8, 8, 8])
    return ExemplarMatcher(
        embedding_model=str(config.get("embedding_model", "color_hist")),
        histogram_bins=(int(bins[0]), int(bins[1]), int(bins[2])),
        score_aggregation=config.get("score_aggregation", "topk_average"),
        top_k=int(config.get("top_k", 3)),
        threshold=float(config.get("threshold", 0.72)),
        nms_iou_threshold=float(config.get("nms_iou_threshold", 0.55)),
    )
