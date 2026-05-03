"""Detection evaluation metrics using pycocotools."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("zerowaste")


def compute_map(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute mAP using pycocotools COCO evaluation.

    Args:
        predictions: List of prediction dicts, each with keys:
            ``image_id``, ``category_id``, ``bbox`` (COCO format), ``score``.
        ground_truths: COCO-format dict with ``images``, ``annotations``, ``categories``.
        iou_thresholds: Custom IoU thresholds. Defaults to COCO standard (0.5:0.95).

    Returns:
        Dict with ``"mAP_50"``, ``"mAP_50_95"``, and per-IoU keys.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Create ground truth COCO object
    coco_gt = COCO()
    coco_gt.dataset = ground_truths
    coco_gt.createIndex()

    if not predictions:
        logger.warning("No predictions provided for evaluation.")
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}

    # Create predictions COCO object
    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "mAP_50_95": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
    }

    return results


def per_class_ap(
    predictions: list[dict[str, Any]],
    ground_truths: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[int, float]:
    """Compute per-class AP at a specific IoU threshold.

    Args:
        predictions: List of prediction dicts.
        ground_truths: COCO-format dict.
        iou_threshold: IoU threshold for AP computation.

    Returns:
        Dict mapping category_id to AP value.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truths
    coco_gt.createIndex()

    if not predictions:
        return {cat_id: 0.0 for cat_id in coco_gt.getCatIds()}

    coco_dt = coco_gt.loadRes(predictions)

    results: dict[int, float] = {}
    for cat_id in coco_gt.getCatIds():
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = np.array([iou_threshold])
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Extract AP for this category
        precision = coco_eval.eval["precision"]
        if precision.size > 0:
            # precision shape: [T, R, K, A, M] — average over recall thresholds
            ap = np.mean(precision[0, :, 0, 0, -1])
            results[cat_id] = float(ap) if ap >= 0 else 0.0
        else:
            results[cat_id] = 0.0

    return results


def build_confusion_matrix(
    predictions: list[dict[str, Any]],
    ground_truths: dict[str, Any],
    num_classes: int,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Build a confusion matrix from detections matched to ground truth via IoU.

    Args:
        predictions: List of prediction dicts.
        ground_truths: COCO-format dict.
        num_classes: Number of object classes.
        iou_threshold: IoU threshold for matching.
        conf_threshold: Minimum confidence for predictions.

    Returns:
        Confusion matrix of shape ``(num_classes + 1, num_classes + 1)``.
        Last row/column = background (false positives / missed detections).
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    # Build category ID to index mapping
    cat_ids = sorted(set(ann["category_id"] for ann in ground_truths["annotations"]))
    cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    # Group by image
    gt_by_img: dict[int, list] = {}
    for ann in ground_truths["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    pred_by_img: dict[int, list] = {}
    for pred in predictions:
        if pred["score"] >= conf_threshold:
            pred_by_img.setdefault(pred["image_id"], []).append(pred)

    all_img_ids = set(list(gt_by_img.keys()) + list(pred_by_img.keys()))

    for img_id in all_img_ids:
        gts = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        matched_gt = set()

        # Sort predictions by confidence (descending)
        preds = sorted(preds, key=lambda p: p["score"], reverse=True)

        for pred in preds:
            pred_cls = cat_to_idx.get(pred["category_id"], num_classes)
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = _compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cls = cat_to_idx.get(gts[best_gt_idx]["category_id"], num_classes)
                matrix[gt_cls, pred_cls] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive
                matrix[num_classes, pred_cls] += 1

        # Missed detections
        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt:
                gt_cls = cat_to_idx.get(gt["category_id"], num_classes)
                matrix[gt_cls, num_classes] += 1

    return matrix


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0
