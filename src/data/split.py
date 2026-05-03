"""Create deterministic labelled/unlabelled splits from COCO annotations."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def create_labelled_unlabelled_split(
    coco_data: dict[str, Any],
    labelled_fraction: float,
    seed: int = 42,
    stratified: bool = True,
) -> tuple[list[int], list[int]]:
    """Split image IDs into labelled and unlabelled subsets.

    Args:
        coco_data: Full COCO dict with ``images`` and ``annotations``.
        labelled_fraction: Fraction of images to label (e.g. 0.1 for 10%).
        seed: Random seed for reproducibility.
        stratified: If True, maintain approximate class balance across splits by
            assigning each image to its dominant category for stratification.

    Returns:
        Tuple of (labelled_image_ids, unlabelled_image_ids).
    """
    if labelled_fraction >= 1.0:
        all_ids = [img["id"] for img in coco_data["images"]]
        return all_ids, []

    if labelled_fraction <= 0.0:
        all_ids = [img["id"] for img in coco_data["images"]]
        return [], all_ids

    rng = random.Random(seed)

    if stratified:
        return _stratified_split(coco_data, labelled_fraction, rng)
    else:
        return _random_split(coco_data, labelled_fraction, rng)


def _random_split(
    coco_data: dict[str, Any],
    labelled_fraction: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Uniform random split."""
    image_ids = [img["id"] for img in coco_data["images"]]
    rng.shuffle(image_ids)
    n_labelled = max(1, int(len(image_ids) * labelled_fraction))
    return image_ids[:n_labelled], image_ids[n_labelled:]


def _stratified_split(
    coco_data: dict[str, Any],
    labelled_fraction: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Stratified split — assign each image to its most frequent category,
    then sample proportionally within each stratum."""

    # Count category occurrences per image
    img_cat_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ann in coco_data["annotations"]:
        img_cat_counts[ann["image_id"]][ann["category_id"]] += 1

    # Assign dominant category per image
    strata: dict[int, list[int]] = defaultdict(list)
    unassigned: list[int] = []

    for img in coco_data["images"]:
        img_id = img["id"]
        if img_id in img_cat_counts:
            dominant = max(img_cat_counts[img_id], key=img_cat_counts[img_id].get)  # type: ignore[arg-type]
            strata[dominant].append(img_id)
        else:
            unassigned.append(img_id)

    labelled: list[int] = []
    unlabelled: list[int] = []

    # Sample from each stratum
    for cat_id in sorted(strata.keys()):
        ids = strata[cat_id]
        rng.shuffle(ids)
        n = max(1, int(len(ids) * labelled_fraction))
        labelled.extend(ids[:n])
        unlabelled.extend(ids[n:])

    # Handle images with no annotations
    rng.shuffle(unassigned)
    n_unassigned = max(0, int(len(unassigned) * labelled_fraction))
    labelled.extend(unassigned[:n_unassigned])
    unlabelled.extend(unassigned[n_unassigned:])

    return labelled, unlabelled


def save_split(
    labelled_ids: list[int],
    unlabelled_ids: list[int],
    output_dir: str | Path,
    labelled_fraction: float,
    seed: int,
) -> Path:
    """Save a split to disk as a JSON file.

    Args:
        labelled_ids: Labelled image IDs.
        unlabelled_ids: Unlabelled image IDs.
        output_dir: Directory to save the split file.
        labelled_fraction: Fraction used (for the filename).
        seed: Seed used (for the filename).

    Returns:
        Path to the saved split file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = output_dir / f"split_frac{labelled_fraction}_seed{seed}.json"
    split_data = {
        "labelled_fraction": labelled_fraction,
        "seed": seed,
        "num_labelled": len(labelled_ids),
        "num_unlabelled": len(unlabelled_ids),
        "labelled_image_ids": labelled_ids,
        "unlabelled_image_ids": unlabelled_ids,
    }
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    return split_file


def load_split(split_file: str | Path) -> tuple[list[int], list[int]]:
    """Load a previously saved split.

    Args:
        split_file: Path to the split JSON file.

    Returns:
        Tuple of (labelled_image_ids, unlabelled_image_ids).
    """
    with open(split_file) as f:
        data = json.load(f)
    return data["labelled_image_ids"], data["unlabelled_image_ids"]
