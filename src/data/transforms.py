"""Albumentations-based augmentation pipelines for detection."""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_train_transforms(cfg: DictConfig) -> A.Compose:
    """Build training augmentation pipeline from config.

    Args:
        cfg: Full experiment config (reads ``augmentation`` and ``model.img_size``).

    Returns:
        Albumentations Compose with bbox support.
    """
    img_size = cfg.model.img_size
    aug = cfg.augmentation

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            A.HorizontalFlip(p=aug.flip_lr),
            A.VerticalFlip(p=aug.flip_ud),
            A.HueSaturationValue(
                hue_shift_limit=int(aug.hsv_h * 180),
                sat_shift_limit=int(aug.hsv_s * 255),
                val_shift_limit=int(aug.hsv_v * 255),
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(1.0 - aug.scale, 1.0 + aug.scale),
                ratio=(0.75, 1.333),
                p=0.5,
            ),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",  # [x_min, y_min, x_max, y_max] normalised
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def get_val_transforms(cfg: DictConfig) -> A.Compose:
    """Build validation/test augmentation pipeline (deterministic resize only).

    Args:
        cfg: Full experiment config.

    Returns:
        Albumentations Compose with bbox support.
    """
    img_size = cfg.model.img_size

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )


def get_domain_shift_transforms(cfg: DictConfig, severity: str = "moderate") -> A.Compose:
    """Build domain-shift augmentations to simulate cross-facility deployment.

    Args:
        cfg: Full experiment config.
        severity: One of ``"mild"``, ``"moderate"``, ``"severe"``.

    Returns:
        Albumentations Compose with bbox support.
    """
    img_size = cfg.model.img_size

    severity_params: dict[str, dict[str, Any]] = {
        "mild": {"brightness": 0.1, "contrast": 0.1, "blur_limit": 3},
        "moderate": {"brightness": 0.2, "contrast": 0.2, "blur_limit": 7},
        "severe": {"brightness": 0.3, "contrast": 0.3, "blur_limit": 11},
    }
    p = severity_params[severity]

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                value=(114, 114, 114),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=p["brightness"],
                contrast_limit=p["contrast"],
                p=1.0,
            ),
            A.GaussianBlur(blur_limit=p["blur_limit"], p=0.5),
            A.ColorJitter(brightness=p["brightness"], contrast=p["contrast"], p=0.5),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )
