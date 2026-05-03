"""ZeroWaste dataset loader — COCO-format bounding box detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ZeroWasteDataset(Dataset):
    """Dataset for ZeroWaste COCO-format object detection.

    Args:
        data_dir: Root directory of a split (e.g. ``data/zerowaste/train``).
        transform: Albumentations transform pipeline (must accept ``image``,
            ``bboxes``, ``class_labels`` keys).
        image_ids: Optional subset of image IDs to load (for labelled/unlabelled splits).
        strip_labels: If True, return empty annotations (for unlabelled pool).
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Any | None = None,
        image_ids: list[int] | None = None,
        strip_labels: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.strip_labels = strip_labels

        ann_file = self.data_dir / "labels.json"
        with open(ann_file) as f:
            coco = json.load(f)

        # Build lookup structures
        self.images: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        self.categories: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco["categories"]
        }

        # Group annotations by image
        self.ann_by_image: dict[int, list[dict]] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.ann_by_image:
                self.ann_by_image[img_id].append(ann)

        # Filter to requested subset
        if image_ids is not None:
            self.image_ids = [iid for iid in image_ids if iid in self.images]
        else:
            self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_dir / "data" / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Build bounding boxes and labels
        if self.strip_labels:
            bboxes: list[list[float]] = []
            class_labels: list[int] = []
        else:
            bboxes = []
            class_labels = []
            for ann in self.ann_by_image[img_id]:
                x, y, bw, bh = ann["bbox"]  # COCO format: x, y, w, h
                # Convert to [x_min, y_min, x_max, y_max] normalised
                x_min = x / w
                y_min = y / h
                x_max = (x + bw) / w
                y_max = (y + bh) / h
                # Clip to [0, 1]
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                x_max = max(0.0, min(1.0, x_max))
                y_max = max(0.0, min(1.0, y_max))
                if x_max > x_min and y_max > y_min:
                    bboxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(ann["category_id"])

        # Apply augmentations
        if self.transform is not None and len(bboxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
        elif self.transform is not None:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed["image"]

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        targets = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            "labels": torch.tensor(class_labels, dtype=torch.long) if class_labels else torch.zeros((0,), dtype=torch.long),
            "image_id": img_id,
        }

        return {"image": image, "targets": targets}

    def get_class_distribution(self) -> dict[int, int]:
        """Count annotations per category across the subset.

        Returns:
            Dict mapping category ID to annotation count.
        """
        counts: dict[int, int] = {}
        for img_id in self.image_ids:
            for ann in self.ann_by_image[img_id]:
                cat = ann["category_id"]
                counts[cat] = counts.get(cat, 0) + 1
        return counts
