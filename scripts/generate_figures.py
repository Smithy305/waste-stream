#!/usr/bin/env python3
"""Generate publication-quality figures from experiment results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style() -> None:
    """Apply consistent publication-ready styling."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def load_results(results_dir: Path, pattern: str) -> dict[str, dict]:
    """Load eval_results.json from matching experiment directories.

    Args:
        results_dir: Top-level results directory.
        pattern: Glob pattern for experiment subdirectories.

    Returns:
        Dict mapping experiment name to results dict.
    """
    results = {}
    for d in sorted(results_dir.glob(pattern)):
        results_file = d / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results[d.name] = json.load(f)
    return results


def plot_map_vs_labelled_fraction(results_dir: Path, output_dir: Path) -> None:
    """mAP vs labelled fraction — supervised vs semi-supervised."""
    fractions = [0.05, 0.1, 0.2, 0.5, 1.0]

    sup_map = []
    semi_map = []

    for frac in fractions:
        sup_file = results_dir / f"supervised_frac{frac}" / "eval_results.json"
        if sup_file.exists():
            with open(sup_file) as f:
                sup_map.append(json.load(f).get("mAP_50", 0))
        else:
            sup_map.append(None)

        semi_file = results_dir / f"semi_sup_frac{frac}" / "eval_results.json"
        if semi_file.exists():
            with open(semi_file) as f:
                semi_map.append(json.load(f).get("mAP_50", 0))
        else:
            semi_map.append(None)

    fig, ax = plt.subplots(figsize=(8, 5))

    valid_sup = [(f, m) for f, m in zip(fractions, sup_map) if m is not None]
    valid_semi = [(f, m) for f, m in zip(fractions, semi_map) if m is not None]

    if valid_sup:
        ax.plot(*zip(*valid_sup), "o-", linewidth=2, markersize=8, label="Supervised")
    if valid_semi:
        ax.plot(*zip(*valid_semi), "s--", linewidth=2, markersize=8, label="Semi-Supervised")

    ax.set_xlabel("Labelled Fraction")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP vs Labelled Fraction")
    ax.set_xscale("log")
    ax.set_xticks(fractions)
    ax.set_xticklabels([str(f) for f in fractions])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "map_vs_labelled_fraction.png")
    plt.close(fig)


def plot_map_vs_noise_rate(results_dir: Path, output_dir: Path) -> None:
    """mAP vs noise rate — comparing loss functions."""
    rates = [0.0, 0.1, 0.2, 0.3]
    loss_fns = {
        "ce": "Cross Entropy",
        "sce": "Symmetric CE",
        "gce": "Generalised CE",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o-", "s--", "^:"]

    for (suffix, label), marker in zip(loss_fns.items(), markers):
        maps = []
        valid_rates = []
        for rate in rates:
            f = results_dir / f"noise_cf_{rate}_{suffix}" / "eval_results.json"
            if f.exists():
                with open(f) as fh:
                    maps.append(json.load(fh).get("mAP_50", 0))
                valid_rates.append(rate)
        if maps:
            ax.plot(valid_rates, maps, marker, linewidth=2, markersize=8, label=label)

    ax.set_xlabel("Noise Rate")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP vs Label Noise Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "map_vs_noise_rate.png")
    plt.close(fig)


def plot_per_class_ap(results_dir: Path, output_dir: Path) -> None:
    """Per-class AP comparison — grouped bar chart."""
    experiments = {
        "supervised_frac1.0": "Supervised (100%)",
        "supervised_frac0.1": "Supervised (10%)",
        "semi_sup_frac0.1": "Semi-Sup (10%)",
    }

    class_names = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]
    data: dict[str, list[float]] = {}

    for exp_name, label in experiments.items():
        f = results_dir / exp_name / "eval_results.json"
        if f.exists():
            with open(f) as fh:
                results = json.load(fh)
            per_class = results.get("per_class_ap", {})
            data[label] = [per_class.get(cls, 0) for cls in class_names]

    if not data:
        return

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, values) in enumerate(data.items()):
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel("Class")
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", " ").title() for c in class_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "per_class_ap.png")
    plt.close(fig)


def plot_pseudo_label_threshold(results_dir: Path, output_dir: Path) -> None:
    """mAP vs pseudo-label confidence threshold."""
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    maps = []
    valid_thresholds = []

    for t in thresholds:
        f = results_dir / f"pl_thresh_{t}" / "eval_results.json"
        if f.exists():
            with open(f) as fh:
                maps.append(json.load(fh).get("mAP_50", 0))
            valid_thresholds.append(t)

    if not maps:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(valid_thresholds, maps, "o-", linewidth=2, markersize=8, color=sns.color_palette()[0])
    ax.set_xlabel("Pseudo-Label Confidence Threshold")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Effect of Pseudo-Label Threshold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "pseudo_label_threshold.png")
    plt.close(fig)


def main() -> None:
    set_style()
    results_dir = Path("results")
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    plot_map_vs_labelled_fraction(results_dir, output_dir)
    print("  - map_vs_labelled_fraction.png")

    plot_map_vs_noise_rate(results_dir, output_dir)
    print("  - map_vs_noise_rate.png")

    plot_per_class_ap(results_dir, output_dir)
    print("  - per_class_ap.png")

    plot_pseudo_label_threshold(results_dir, output_dir)
    print("  - pseudo_label_threshold.png")

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
