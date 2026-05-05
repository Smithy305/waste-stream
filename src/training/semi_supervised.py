"""Teacher-student semi-supervised training loop with pseudo-labelling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig

from src.data.split import create_labelled_unlabelled_split, load_split, save_split
from src.models.detector import Detector
from src.models.ema import EMATeacher
from src.utils.seed import seed_everything

logger = logging.getLogger("zerowaste")


class SemiSupervisedTrainer:
    """Teacher-student semi-supervised training with EMA pseudo-labelling.

    The pipeline:
      1. Train the student on labelled data for a warmup period.
      2. Initialise the EMA teacher from the student.
      3. Teacher generates pseudo-labels on the unlabelled pool.
      4. Student trains on labelled + pseudo-labelled data.
      5. Teacher is updated as EMA of student after each epoch.
      6. Repeat from step 3.

    Args:
        cfg: Full experiment config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.student = Detector(cfg)
        self.ss_cfg = cfg.semi_supervised

    def prepare_splits(self) -> tuple[list[int], list[int]]:
        """Create or load the labelled/unlabelled split.

        Returns:
            Tuple of (labelled_ids, unlabelled_ids).
        """
        data_dir = Path(self.cfg.data_dir)
        ann_file = data_dir / "train" / "labels.json"

        with open(ann_file) as f:
            coco_data = json.load(f)

        labelled_fraction = self.cfg.get("labelled_fraction", 0.1)
        split_dir = data_dir / "splits"
        split_file = split_dir / f"split_frac{labelled_fraction}_seed{self.cfg.seed}.json"

        if split_file.exists():
            labelled_ids, unlabelled_ids = load_split(split_file)
            logger.info("Loaded split: %d labelled, %d unlabelled", len(labelled_ids), len(unlabelled_ids))
        else:
            labelled_ids, unlabelled_ids = create_labelled_unlabelled_split(
                coco_data, labelled_fraction, seed=self.cfg.seed
            )
            save_split(labelled_ids, unlabelled_ids, split_dir, labelled_fraction, self.cfg.seed)
            logger.info("Created split: %d labelled, %d unlabelled", len(labelled_ids), len(unlabelled_ids))

        return labelled_ids, unlabelled_ids

    def generate_pseudo_labels(
        self,
        teacher: EMATeacher,
        unlabelled_ids: list[int],
        coco_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the teacher model to generate pseudo-labels on unlabelled images.

        Args:
            teacher: EMA teacher model.
            unlabelled_ids: Image IDs in the unlabelled pool.
            coco_data: Full COCO dict for image path resolution.

        Returns:
            COCO-format dict containing only pseudo-labelled images and annotations.
        """
        data_dir = Path(self.cfg.data_dir) / "train" / "data"

        img_lookup = {img["id"]: img for img in coco_data["images"]}
        cat_ids = sorted(cat["id"] for cat in coco_data["categories"])

        # Resolve per-class thresholds. Falls back to global pl_threshold for
        # any category not explicitly listed in cfg.semi_supervised.class_thresholds.
        global_threshold = float(self.ss_cfg.pl_threshold)
        class_thresholds_cfg = self.ss_cfg.get("class_thresholds", None)
        class_thresholds = {cat_id: global_threshold for cat_id in cat_ids}
        if class_thresholds_cfg is not None:
            for cat_id_str, val in dict(class_thresholds_cfg).items():
                class_thresholds[int(cat_id_str)] = float(val)
        # Lowest threshold drives the predict() call so we get all candidates
        min_threshold = min(class_thresholds.values())

        # Resolve per-class caps. Same fallback semantics as thresholds.
        global_cap = self.ss_cfg.get("max_pseudo_per_class", None)
        per_class_caps_cfg = self.ss_cfg.get("max_pseudo_per_class_dict", None)
        class_caps: dict[int, int | None] = {cat_id: global_cap for cat_id in cat_ids}
        if per_class_caps_cfg is not None:
            for cat_id_str, val in dict(per_class_caps_cfg).items():
                class_caps[int(cat_id_str)] = int(val)

        # Collect candidates per (img_id, class), then post-process for caps.
        # Each candidate: (img_id, img_info, cat_id, bbox, confidence)
        candidates: list[tuple[int, dict, int, list[float], float]] = []

        for img_id in unlabelled_ids:
            if img_id not in img_lookup:
                continue

            img_info = img_lookup[img_id]
            img_path = data_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            # Run teacher inference at the lowest per-class threshold so we
            # don't pre-filter minority classes.
            results = self.student.predict(
                source=str(img_path),
                conf=min_threshold,
            )

            if not results or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Map YOLO class index back to category ID
                cat_id = cat_ids[cls] if cls < len(cat_ids) else cat_ids[0]

                # Apply per-class threshold
                if conf < class_thresholds[cat_id]:
                    continue

                x1, y1, x2, y2 = xyxy
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                candidates.append((img_id, img_info, cat_id, bbox, conf))

        # Apply per-class caps: keep top-N highest-confidence detections per class.
        kept_by_class: dict[int, list[tuple[int, dict, int, list[float], float]]] = {}
        for cand in candidates:
            kept_by_class.setdefault(cand[2], []).append(cand)

        kept: list[tuple[int, dict, int, list[float], float]] = []
        for cat_id, group in kept_by_class.items():
            group.sort(key=lambda c: c[4], reverse=True)
            cap = class_caps.get(cat_id)
            if cap is not None and cap > 0:
                group = group[:cap]
            kept.extend(group)

        # Build COCO output
        pseudo_images_set: dict[int, dict] = {}
        pseudo_annotations: list[dict] = []
        ann_id = 0
        class_counts: dict[str, int] = {}
        total_conf = 0.0

        for img_id, img_info, cat_id, bbox, conf in kept:
            pseudo_images_set[img_id] = img_info
            pseudo_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "confidence": conf,
            })
            ann_id += 1
            class_counts[str(cat_id)] = class_counts.get(str(cat_id), 0) + 1
            total_conf += conf

        pseudo_images = list(pseudo_images_set.values())
        mean_conf = total_conf / len(kept) if kept else 0.0

        logger.info(
            "Pseudo-labels: %d candidates → %d kept after caps (%d images, mean conf=%.3f)",
            len(candidates), len(kept), len(pseudo_images), mean_conf,
        )
        for cls_name, count in sorted(class_counts.items()):
            cap = class_caps.get(int(cls_name))
            cap_str = f" (cap={cap})" if cap else ""
            logger.info("  Class %s: %d pseudo-labels%s", cls_name, count, cap_str)

        return {
            "images": pseudo_images,
            "annotations": pseudo_annotations,
            "categories": coco_data["categories"],
        }

    def train(self) -> Any:
        """Run the full semi-supervised training pipeline.

        Returns:
            Final training results.
        """
        data_dir = Path(self.cfg.data_dir)
        labelled_ids, unlabelled_ids = self.prepare_splits()

        ann_file = data_dir / "train" / "labels.json"
        with open(ann_file) as f:
            coco_data = json.load(f)

        # --- Phase 1: Supervised warmup on labelled data ---
        warmup_epochs = self.ss_cfg.teacher_warmup_epochs
        logger.info("Phase 1: Supervised warmup for %d epochs", warmup_epochs)

        work_dir = Path(self.cfg.output_dir) / self.cfg.get("experiment_name", "semi_supervised")
        work_dir.mkdir(parents=True, exist_ok=True)

        labelled_coco = self._filter_coco(coco_data, labelled_ids)
        data_yaml = self._write_yolo_data(labelled_coco, coco_data, data_dir, work_dir, "warmup")

        self.student.train_model(
            data_yaml=data_yaml,
            epochs=warmup_epochs,
            project=str(work_dir),
            name="warmup",
        )

        # Track the latest best checkpoint so each round can reload cleanly
        # (Ultralytics doesn't support calling .train() twice on the same YOLO instance).
        latest_ckpt = self._find_best_checkpoint(work_dir, "warmup")
        self.student.load_checkpoint(latest_ckpt)

        # --- Phase 2: Initialise EMA teacher ---
        logger.info("Phase 2: Initialising EMA teacher (decay=%.6f)", self.ss_cfg.ema_decay)
        teacher = EMATeacher(self.student.model.model, decay=self.ss_cfg.ema_decay)

        # --- Phase 3: Iterative pseudo-labelling ---
        remaining_epochs = self.cfg.training.epochs - warmup_epochs
        num_rounds = max(1, remaining_epochs // 10)
        epochs_per_round = remaining_epochs // num_rounds

        logger.info(
            "Phase 3: Semi-supervised training — %d rounds of %d epochs",
            num_rounds, epochs_per_round,
        )

        results = None
        for round_idx in range(num_rounds):
            logger.info("=== Round %d/%d ===", round_idx + 1, num_rounds)

            # Generate pseudo-labels
            pseudo_coco = self.generate_pseudo_labels(teacher, unlabelled_ids, coco_data)

            # Merge labelled + pseudo-labelled data
            merged_coco = self._merge_coco(labelled_coco, pseudo_coco)

            # Write merged data
            data_yaml = self._write_yolo_data(
                merged_coco, coco_data, data_dir, work_dir, f"round{round_idx}"
            )

            # Train student with retry on the known MPS task-aligned-assigner crash.
            # That bug is non-deterministic — reloading from the same checkpoint and
            # retrying with a smaller batch size almost always gets past it.
            results = self._train_round_with_retry(
                data_yaml=data_yaml,
                epochs=epochs_per_round,
                work_dir=work_dir,
                run_name=f"round{round_idx}",
                source_ckpt=latest_ckpt,
            )

            # Update latest checkpoint pointer for the next round
            latest_ckpt = self._find_best_checkpoint(work_dir, f"round{round_idx}")

            # Update teacher EMA
            teacher.update(self.student.model.model)
            logger.info("Updated EMA teacher (total updates: %d)", teacher.num_updates)

            # Save teacher checkpoint
            teacher_ckpt = work_dir / f"teacher_round{round_idx}.pt"
            torch.save(teacher.state_dict(), teacher_ckpt)

        logger.info("Semi-supervised training complete.")
        return results

    def _find_best_checkpoint(self, work_dir: Path, run_name: str) -> Path:
        """Locate the best.pt written by Ultralytics for a given run.

        Ultralytics may write to either ``<work_dir>/<run_name>/weights/best.pt``
        or ``runs/detect/<work_dir>/<run_name>/weights/best.pt`` depending on version
        and project path resolution. Search both.
        """
        candidates = [
            work_dir / run_name / "weights" / "best.pt",
            Path("runs/detect") / work_dir / run_name / "weights" / "best.pt",
        ]
        for ckpt in candidates:
            if ckpt.exists():
                return ckpt

        # Fall back to a glob search for any best.pt under work_dir or runs/detect
        for root in [work_dir, Path("runs/detect")]:
            matches = sorted(root.rglob(f"{run_name}/weights/best.pt"))
            if matches:
                return matches[-1]

        raise FileNotFoundError(
            f"Could not find best.pt for run '{run_name}' under {work_dir} or runs/detect/"
        )

    def _train_round_with_retry(
        self,
        data_yaml: Path,
        epochs: int,
        work_dir: Path,
        run_name: str,
        source_ckpt: Path,
        max_retries: int = 2,
    ) -> Any:
        """Run a training round with retry on the MPS shape-mismatch crash.

        The Ultralytics task-aligned assigner occasionally crashes on MPS with
        a non-deterministic shape mismatch error. Retrying from the same
        checkpoint with a halved batch size usually succeeds.
        """
        original_batch = self.cfg.training.batch_size
        attempt = 0

        while True:
            # Always reload from the source checkpoint — Ultralytics doesn't allow
            # calling .train() twice on the same YOLO instance.
            self.student.load_checkpoint(source_ckpt)

            try:
                batch_override = max(2, original_batch // (2 ** attempt))
                if attempt > 0:
                    logger.warning(
                        "Retry %d/%d for %s — reducing batch size to %d",
                        attempt, max_retries, run_name, batch_override,
                    )
                return self.student.train_model(
                    data_yaml=data_yaml,
                    epochs=epochs,
                    batch_size=batch_override,
                    project=str(work_dir),
                    name=f"{run_name}_retry{attempt}" if attempt > 0 else run_name,
                )
            except RuntimeError as e:
                msg = str(e)
                # Recognise the MPS assigner shape mismatch specifically.
                is_mps_assigner_bug = (
                    "shape mismatch" in msg
                    and "broadcast to indexing result" in msg
                )
                if not is_mps_assigner_bug or attempt >= max_retries:
                    raise
                logger.warning("Caught MPS assigner crash on %s: %s", run_name, msg)
                attempt += 1

    def _filter_coco(self, coco_data: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
        """Filter COCO data to a subset of image IDs."""
        id_set = set(image_ids)
        return {
            "images": [img for img in coco_data["images"] if img["id"] in id_set],
            "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in id_set],
            "categories": coco_data["categories"],
        }

    def _merge_coco(self, labelled: dict[str, Any], pseudo: dict[str, Any]) -> dict[str, Any]:
        """Merge labelled and pseudo-labelled COCO dicts."""
        # Re-number pseudo annotation IDs to avoid collisions
        max_ann_id = max((ann["id"] for ann in labelled["annotations"]), default=0) + 1
        pseudo_anns = []
        for ann in pseudo["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = max_ann_id
            max_ann_id += 1
            pseudo_anns.append(new_ann)

        return {
            "images": labelled["images"] + pseudo["images"],
            "annotations": labelled["annotations"] + pseudo_anns,
            "categories": labelled["categories"],
        }

    def _write_yolo_data(
        self,
        train_coco: dict[str, Any],
        full_coco: dict[str, Any],
        source_dir: Path,
        work_dir: Path,
        stage: str,
    ) -> Path:
        """Convert COCO data to YOLO format and write data YAML.

        Args:
            train_coco: Training COCO dict (labelled or merged).
            full_coco: Full COCO dict (for val set).
            source_dir: Root dataset dir.
            work_dir: Experiment work directory.
            stage: Stage name for subdirectory.

        Returns:
            Path to the YOLO data YAML.
        """
        yolo_dir = work_dir / f"yolo_data_{stage}"
        self._convert_coco_to_yolo(train_coco, source_dir, yolo_dir, "train")

        # Val set
        val_ann_file = source_dir / "val" / "labels.json"
        if val_ann_file.exists():
            with open(val_ann_file) as f:
                val_coco = json.load(f)
            self._convert_coco_to_yolo(val_coco, source_dir, yolo_dir, "val")

        data_yaml = work_dir / f"data_{stage}.yaml"
        yolo_config = {
            "path": str(yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": self.cfg.dataset.num_classes,
            "names": list(self.cfg.dataset.class_names),
        }
        with open(data_yaml, "w") as f:
            yaml.dump(yolo_config, f, default_flow_style=False)

        return data_yaml

    def _convert_coco_to_yolo(
        self,
        coco_data: dict[str, Any],
        source_data_dir: Path,
        output_dir: Path,
        split: str,
    ) -> None:
        """Convert COCO annotations to YOLO txt format with symlinked images."""
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_lookup = {img["id"]: img for img in coco_data["images"]}
        anns_by_img: dict[int, list] = {img_id: [] for img_id in img_lookup}
        for ann in coco_data["annotations"]:
            if ann["image_id"] in anns_by_img:
                anns_by_img[ann["image_id"]].append(ann)

        cat_ids = sorted(cat["id"] for cat in coco_data["categories"])
        cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        for img_id, img_info in img_lookup.items():
            src_img = source_data_dir / split / "data" / img_info["file_name"]
            dst_img = img_dir / img_info["file_name"]
            if not dst_img.exists() and src_img.exists():
                dst_img.symlink_to(src_img.resolve())

            w, h = img_info["width"], img_info["height"]
            label_name = Path(img_info["file_name"]).stem + ".txt"
            label_path = lbl_dir / label_name

            lines = []
            for ann in anns_by_img[img_id]:
                cls_idx = cat_to_idx.get(ann["category_id"], 0)
                bx, by, bw, bh = ann["bbox"]
                x_center = (bx + bw / 2) / w
                y_center = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
