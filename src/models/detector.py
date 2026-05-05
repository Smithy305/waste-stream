"""Wrapper around Ultralytics YOLOv8 for detection with a consistent API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from ultralytics import YOLO

logger = logging.getLogger("zerowaste")


class Detector(nn.Module):
    """YOLOv8 detection wrapper exposing train/predict/export via a unified interface.

    This wraps the Ultralytics Python API so that the rest of the pipeline interacts
    through a stable interface, making it easy to swap backbones later (e.g. RT-DETR).

    Args:
        cfg: Full experiment config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_name = cfg.model.backbone  # e.g. "yolov8s"
        pretrained = cfg.model.get("pretrained", True)

        if pretrained:
            self.model = YOLO(f"{model_name}.pt")
            logger.info("Loaded pretrained %s", model_name)
        else:
            self.model = YOLO(f"{model_name}.yaml")
            logger.info("Initialised %s from scratch", model_name)

        self.num_classes = cfg.dataset.num_classes

    def train_model(
        self,
        data_yaml: str | Path,
        epochs: int | None = None,
        batch_size: int | None = None,
        project: str | None = None,
        name: str | None = None,
        resume: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run Ultralytics training.

        Args:
            data_yaml: Path to YOLO-format data config YAML.
            epochs: Override training epochs.
            batch_size: Override batch size.
            project: Output project directory.
            name: Run name within the project.
            resume: Resume from last checkpoint.
            **kwargs: Additional Ultralytics training args.

        Returns:
            Ultralytics training results.
        """
        train_cfg = self.cfg.training
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs or train_cfg.epochs,
            batch=batch_size or train_cfg.batch_size,
            imgsz=self.cfg.model.img_size,
            lr0=train_cfg.lr,
            optimizer=train_cfg.optimizer,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
            warmup_epochs=train_cfg.warmup_epochs,
            patience=train_cfg.patience,
            workers=train_cfg.num_workers,
            project=project or self.cfg.output_dir,
            name=name or self.cfg.get("experiment_name", "train"),
            device=train_cfg.device if train_cfg.device else None,
            resume=resume,
            verbose=True,
            conf=0.25,
            max_det=100,
            **kwargs,
        )

    @torch.no_grad()
    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        **kwargs: Any,
    ) -> list[Any]:
        """Run inference on images.

        Args:
            source: Image path, directory, tensor, or numpy array.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            max_det: Maximum detections per image.
            **kwargs: Additional predict args.

        Returns:
            List of Ultralytics Results objects.
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            **kwargs,
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export model to a deployment format.

        Args:
            format: Export format (e.g. ``"onnx"``, ``"torchscript"``).
            **kwargs: Additional export args.

        Returns:
            Path to the exported model file.
        """
        return self.model.export(format=format, **kwargs)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint.

        Args:
            path: Path to ``.pt`` checkpoint.
        """
        self.model = YOLO(str(path))
        logger.info("Loaded checkpoint from %s", path)

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the underlying model's state dict."""
        return self.model.model.state_dict()

    def load_state_dict_from(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a state dict into the underlying model."""
        self.model.model.load_state_dict(state_dict)
