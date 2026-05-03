"""Standard supervised training loop using Ultralytics YOLOv8."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from src.data.label_noise import apply_noise
from src.data.split import create_labelled_unlabelled_split, load_split, save_split
from src.models.detector import Detector
from src.utils.seed import seed_everything

logger = logging.getLogger("zerowaste")


class SupervisedTrainer:
    """Supervised training pipeline: handles splits, noise injection, and YOLO training.

    Args:
        cfg: Full experiment config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.detector = Detector(cfg)

    def prepare_data(self) -> Path:
        """Create labelled split, optionally inject noise, and write a YOLO data YAML.

        Returns:
            Path to the YOLO data config YAML.
        """
        data_dir = Path(self.cfg.data_dir)
        train_dir = data_dir / "train"
        ann_file = train_dir / "labels.json"

        with open(ann_file) as f:
            coco_data = json.load(f)

        labelled_fraction = self.cfg.get("labelled_fraction", 1.0)

        # Create or load split
        split_dir = data_dir / "splits"
        split_file = split_dir / f"split_frac{labelled_fraction}_seed{self.cfg.seed}.json"

        if split_file.exists():
            labelled_ids, _ = load_split(split_file)
            logger.info("Loaded existing split from %s (%d labelled images)", split_file, len(labelled_ids))
        else:
            labelled_ids, unlabelled_ids = create_labelled_unlabelled_split(
                coco_data, labelled_fraction, seed=self.cfg.seed
            )
            save_split(labelled_ids, unlabelled_ids, split_dir, labelled_fraction, self.cfg.seed)
            logger.info(
                "Created split: %d labelled, %d unlabelled (frac=%.2f)",
                len(labelled_ids), len(unlabelled_ids), labelled_fraction,
            )

        # Filter annotations to labelled subset
        labelled_set = set(labelled_ids)
        filtered_coco = {
            "images": [img for img in coco_data["images"] if img["id"] in labelled_set],
            "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in labelled_set],
            "categories": coco_data["categories"],
        }

        # Inject noise if configured
        noise_cfg = self.cfg.get("noise", {})
        if noise_cfg.get("enabled", False):
            logger.info(
                "Injecting %s noise at rate %.2f",
                noise_cfg.noise_type, noise_cfg.noise_rate,
            )
            filtered_coco = apply_noise(
                filtered_coco,
                noise_type=noise_cfg.noise_type,
                noise_rate=noise_cfg.noise_rate,
                seed=self.cfg.seed,
                bbox_jitter_sigma=noise_cfg.get("bbox_jitter_sigma", 0.05),
            )

        # Write filtered COCO annotations for YOLO conversion
        work_dir = Path(self.cfg.output_dir) / self.cfg.get("experiment_name", "supervised")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Convert COCO to YOLO format
        yolo_data_dir = work_dir / "yolo_data"
        self._convert_coco_to_yolo(filtered_coco, data_dir, yolo_data_dir, split="train")

        # Also convert val set
        val_ann_file = data_dir / "val" / "labels.json"
        if val_ann_file.exists():
            with open(val_ann_file) as f:
                val_coco = json.load(f)
            self._convert_coco_to_yolo(val_coco, data_dir, yolo_data_dir, split="val")

        # Write YOLO data YAML
        data_yaml = work_dir / "data.yaml"
        yolo_config = {
            "path": str(yolo_data_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": self.cfg.dataset.num_classes,
            "names": list(self.cfg.dataset.class_names),
        }
        with open(data_yaml, "w") as f:
            yaml.dump(yolo_config, f, default_flow_style=False)

        logger.info("YOLO data config written to %s", data_yaml)
        return data_yaml

    def train(self) -> Any:
        """Run the full supervised training pipeline.

        Returns:
            Ultralytics training results.
        """
        data_yaml = self.prepare_data()

        logger.info("Starting supervised training: %s", self.cfg.get("experiment_name", "supervised"))
        results = self.detector.train_model(
            data_yaml=data_yaml,
            project=self.cfg.output_dir,
            name=self.cfg.get("experiment_name", "supervised"),
        )
        logger.info("Supervised training complete.")
        return results

    def _convert_coco_to_yolo(
        self,
        coco_data: dict[str, Any],
        source_data_dir: Path,
        output_dir: Path,
        split: str,
    ) -> None:
        """Convert COCO annotations to YOLO txt format with symlinked images.

        Args:
            coco_data: COCO dict.
            source_data_dir: Root dataset dir (contains train/val/test).
            output_dir: Where to write YOLO-format data.
            split: ``"train"`` or ``"val"``.
        """
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Build image info lookup
        img_lookup = {img["id"]: img for img in coco_data["images"]}

        # Group annotations by image
        anns_by_img: dict[int, list] = {img_id: [] for img_id in img_lookup}
        for ann in coco_data["annotations"]:
            if ann["image_id"] in anns_by_img:
                anns_by_img[ann["image_id"]].append(ann)

        # Build category ID to 0-indexed class mapping
        cat_ids = sorted(cat["id"] for cat in coco_data["categories"])
        cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        for img_id, img_info in img_lookup.items():
            # Symlink image
            src_img = source_data_dir / split / "data" / img_info["file_name"]
            dst_img = img_dir / img_info["file_name"]
            if not dst_img.exists() and src_img.exists():
                dst_img.symlink_to(src_img.resolve())

            # Write YOLO label file
            w, h = img_info["width"], img_info["height"]
            label_name = Path(img_info["file_name"]).stem + ".txt"
            label_path = lbl_dir / label_name

            lines = []
            for ann in anns_by_img[img_id]:
                cls_idx = cat_to_idx[ann["category_id"]]
                bx, by, bw, bh = ann["bbox"]
                # YOLO format: class x_center y_center width height (normalised)
                x_center = (bx + bw / 2) / w
                y_center = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
