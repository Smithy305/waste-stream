"""Inject synthetic label noise into COCO annotations for robustness experiments."""

from __future__ import annotations

import copy
import random
from typing import Any


def inject_class_flip_noise(
    annotations: list[dict[str, Any]],
    category_ids: list[int],
    noise_rate: float = 0.2,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Randomly flip class labels with a given probability.

    Args:
        annotations: List of COCO annotation dicts (modified in-place on copies).
        category_ids: All valid category IDs to flip between.
        noise_rate: Probability of flipping each annotation's class.
        rng: Optional Random instance for reproducibility.

    Returns:
        New list of annotations with noisy class labels.
    """
    rng = rng or random.Random()
    noisy = copy.deepcopy(annotations)
    num_flipped = 0

    for ann in noisy:
        if rng.random() < noise_rate:
            original = ann["category_id"]
            candidates = [c for c in category_ids if c != original]
            if candidates:
                ann["category_id"] = rng.choice(candidates)
                num_flipped += 1

    return noisy


def inject_bbox_jitter(
    annotations: list[dict[str, Any]],
    sigma: float = 0.05,
    img_width: int = 640,
    img_height: int = 640,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Add Gaussian noise to bounding box coordinates.

    Args:
        annotations: List of COCO annotation dicts (``bbox`` in ``[x, y, w, h]`` format).
        sigma: Standard deviation of noise as a fraction of the box dimension.
        img_width: Image width for clipping.
        img_height: Image height for clipping.
        rng: Optional Random instance for reproducibility.

    Returns:
        New list of annotations with jittered bounding boxes.
    """
    rng = rng or random.Random()
    noisy = copy.deepcopy(annotations)

    for ann in noisy:
        x, y, w, h = ann["bbox"]
        dx = rng.gauss(0, sigma * w)
        dy = rng.gauss(0, sigma * h)
        dw = rng.gauss(0, sigma * w)
        dh = rng.gauss(0, sigma * h)

        x_new = max(0, x + dx)
        y_new = max(0, y + dy)
        w_new = max(1, w + dw)
        h_new = max(1, h + dh)

        # Clip to image bounds
        x_new = min(x_new, img_width - 1)
        y_new = min(y_new, img_height - 1)
        w_new = min(w_new, img_width - x_new)
        h_new = min(h_new, img_height - y_new)

        ann["bbox"] = [x_new, y_new, w_new, h_new]

    return noisy


def inject_missing_annotations(
    annotations: list[dict[str, Any]],
    drop_rate: float = 0.2,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Randomly drop annotations to simulate incomplete labelling.

    Args:
        annotations: List of COCO annotation dicts.
        drop_rate: Probability of dropping each annotation.
        rng: Optional Random instance for reproducibility.

    Returns:
        Filtered list of annotations.
    """
    rng = rng or random.Random()
    return [ann for ann in annotations if rng.random() >= drop_rate]


def apply_noise(
    coco_data: dict[str, Any],
    noise_type: str,
    noise_rate: float,
    seed: int = 42,
    bbox_jitter_sigma: float = 0.05,
) -> dict[str, Any]:
    """Apply a noise strategy to an entire COCO annotation file.

    Args:
        coco_data: Full COCO dict with ``images``, ``annotations``, ``categories``.
        noise_type: One of ``"class_flip"``, ``"bbox_jitter"``, ``"missing_annot"``.
        noise_rate: Corruption probability.
        seed: Random seed.
        bbox_jitter_sigma: Sigma for bbox jitter (only used if ``noise_type="bbox_jitter"``).

    Returns:
        New COCO dict with noisy annotations.
    """
    rng = random.Random(seed)
    result = copy.deepcopy(coco_data)
    category_ids = [cat["id"] for cat in result["categories"]]

    # Build image dimension lookup
    img_dims = {img["id"]: (img["width"], img["height"]) for img in result["images"]}

    if noise_type == "class_flip":
        result["annotations"] = inject_class_flip_noise(
            result["annotations"], category_ids, noise_rate, rng
        )
    elif noise_type == "bbox_jitter":
        noisy_anns = []
        for ann in result["annotations"]:
            w, h = img_dims.get(ann["image_id"], (640, 640))
            jittered = inject_bbox_jitter([ann], bbox_jitter_sigma, w, h, rng)
            noisy_anns.extend(jittered)
        result["annotations"] = noisy_anns
    elif noise_type == "missing_annot":
        result["annotations"] = inject_missing_annotations(
            result["annotations"], noise_rate, rng
        )
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return result
