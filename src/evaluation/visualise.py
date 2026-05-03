"""Detection visualisation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Consistent palette for waste categories
CATEGORY_COLOURS = {
    "rigid_plastic": (0, 114, 189),
    "cardboard": (217, 83, 25),
    "metal": (237, 177, 32),
    "soft_plastic": (126, 47, 142),
}

DEFAULT_COLOURS = [
    (0, 114, 189),
    (217, 83, 25),
    (237, 177, 32),
    (126, 47, 142),
    (119, 172, 48),
    (77, 190, 238),
    (162, 20, 47),
]


def draw_detections(
    image: np.ndarray,
    boxes: list[list[float]],
    labels: list[str],
    scores: list[float] | None = None,
    class_names: dict[int, str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR image (H, W, 3).
        boxes: List of ``[x, y, w, h]`` bounding boxes (COCO format).
        labels: List of label strings or category names.
        scores: Optional confidence scores to display.
        class_names: Mapping of class indices to names (for colour lookup).
        thickness: Box line thickness.
        font_scale: Text size.

    Returns:
        Annotated image copy.
    """
    img = image.copy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x, y, w, h = [int(v) for v in box]
        colour = CATEGORY_COLOURS.get(label, DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)])

        cv2.rectangle(img, (x, y), (x + w, y + h), colour, thickness)

        text = label
        if scores is not None and i < len(scores):
            text = f"{label} {scores[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), colour, -1)
        cv2.putText(
            img, text, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    return img


def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    class_name: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a precision-recall curve.

    Args:
        precisions: Array of precision values.
        recalls: Array of recall values.
        class_name: Class name for the title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recalls, precisions, linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall: {class_name}" if class_name else "Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        matrix: Confusion matrix of shape ``(C+1, C+1)`` where last row/col = background.
        class_names: Class names (without background).
        save_path: Optional path to save the figure.
        normalize: If True, normalise rows to sum to 1.

    Returns:
        Matplotlib Figure.
    """
    _set_plot_style()
    labels = class_names + ["background"]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix.astype(np.float64) / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_side_by_side(
    image: np.ndarray,
    gt_boxes: list[list[float]],
    gt_labels: list[str],
    pred_boxes: list[list[float]],
    pred_labels: list[str],
    pred_scores: list[float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ground truth and predictions side by side.

    Args:
        image: BGR image.
        gt_boxes: Ground truth boxes (COCO format).
        gt_labels: Ground truth labels.
        pred_boxes: Predicted boxes.
        pred_labels: Predicted labels.
        pred_scores: Optional confidence scores.
        save_path: Optional path to save.

    Returns:
        Matplotlib Figure.
    """
    _set_plot_style()
    gt_img = draw_detections(image, gt_boxes, gt_labels)
    pred_img = draw_detections(image, pred_boxes, pred_labels, scores=pred_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Predictions")
    axes[1].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _set_plot_style() -> None:
    """Apply consistent publication-ready plot styling."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })
