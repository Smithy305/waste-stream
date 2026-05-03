#!/usr/bin/env python3
"""Entry point: evaluate a trained model on the test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compute_map, per_class_ap, build_confusion_matrix
from src.evaluation.visualise import plot_confusion_matrix, plot_side_by_side
from src.models.detector import Detector
from src.utils.config import load_config, parse_cli_overrides
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main() -> None:
    config_path, overrides = parse_cli_overrides()
    if config_path is None:
        config_path = "configs/supervised_baseline.yaml"

    cfg = load_config(config_path, overrides)
    seed_everything(cfg.seed)

    logger = setup_logger()

    # Load checkpoint
    checkpoint = None
    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                checkpoint = sys.argv[idx + 1]
                break

    if checkpoint is None:
        logger.error("Usage: python evaluate.py --config <yaml> --checkpoint <path>")
        sys.exit(1)

    logger.info("Loading model from %s", checkpoint)
    detector = Detector(cfg)
    detector.load_checkpoint(checkpoint)

    # Load test set
    data_dir = Path(cfg.data_dir)
    test_ann_file = data_dir / "test" / "labels.json"
    if not test_ann_file.exists():
        # Fall back to val set
        test_ann_file = data_dir / "val" / "labels.json"
        logger.warning("No test set found, using val set for evaluation.")

    with open(test_ann_file) as f:
        gt_coco = json.load(f)

    # Run inference on all test images
    test_img_dir = test_ann_file.parent / "data"
    predictions: list[dict[str, Any]] = []
    cat_ids = sorted(cat["id"] for cat in gt_coco["categories"])

    logger.info("Running inference on %d images...", len(gt_coco["images"]))
    for img_info in gt_coco["images"]:
        img_path = test_img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        results = detector.predict(str(img_path), conf=0.001)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                predictions.append({
                    "image_id": img_info["id"],
                    "category_id": cat_ids[int(boxes.cls[i])] if int(boxes.cls[i]) < len(cat_ids) else cat_ids[0],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(boxes.conf[i]),
                })

    logger.info("Generated %d predictions.", len(predictions))

    # Compute metrics
    map_results = compute_map(predictions, gt_coco)
    logger.info("mAP@0.5: %.4f", map_results["mAP_50"])
    logger.info("mAP@0.5:0.95: %.4f", map_results["mAP_50_95"])

    class_ap = per_class_ap(predictions, gt_coco)
    class_names = {cat["id"]: cat["name"] for cat in gt_coco["categories"]}
    for cat_id, ap in class_ap.items():
        logger.info("  %s AP@0.5: %.4f", class_names.get(cat_id, cat_id), ap)

    # Confusion matrix
    cm = build_confusion_matrix(predictions, gt_coco, cfg.dataset.num_classes)
    output_dir = Path(cfg.output_dir) / cfg.get("experiment_name", "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        list(cfg.dataset.class_names),
        save_path=output_dir / "confusion_matrix.png",
    )

    # Save results
    results_file = output_dir / "eval_results.json"
    eval_results = {
        **map_results,
        "per_class_ap": {class_names.get(k, str(k)): v for k, v in class_ap.items()},
    }
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info("Results saved to %s", results_file)


if __name__ == "__main__":
    main()
