# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semi-supervised object detection for waste stream analysis using the [ZeroWaste](https://zenodo.org/records/6412647) dataset (COCO format). Addresses three challenges: semi-supervised learning with limited labels, noisy label robustness, and domain shift across recycling facilities. Portfolio/interview-prep project — code quality and reproducibility matter as much as results.

## Tech Stack

Python 3.10+ · PyTorch 2.x · Ultralytics YOLOv8 (Python API, not CLI) · Albumentations · OmegaConf · Weights & Biases · pycocotools

## Setup

```bash
pip install -r requirements.txt
bash data/download_zerowaste.sh
```

## Commands

```bash
# Supervised baseline (full labels)
python scripts/train_supervised.py --config configs/supervised_baseline.yaml

# Supervised with reduced labels (CLI overrides use OmegaConf dotlist syntax)
python scripts/train_supervised.py --config configs/supervised_baseline.yaml labelled_fraction=0.1

# Semi-supervised teacher-student training
python scripts/train_semi_supervised.py --config configs/semi_supervised.yaml

# Noisy label experiment
python scripts/train_supervised.py --config configs/noisy_labels.yaml noise.noise_rate=0.2

# Evaluation
python scripts/evaluate.py --config configs/semi_supervised.yaml --checkpoint results/best.pt

# Run all ablation experiments
bash scripts/run_ablations.sh

# Generate publication-quality figures from results
python scripts/generate_figures.py
```

## Architecture

**Config system**: All configs inherit from `configs/base.yaml` via a `base:` key. CLI overrides use OmegaConf dotlist syntax (`key.subkey=value`). Config loading is in `src/utils/config.py`.

**Data pipeline**: COCO-format annotations → `ZeroWasteDataset` (src/data/zerowaste.py) loads images + boxes with normalised coordinates. `src/data/split.py` creates deterministic stratified labelled/unlabelled splits saved as JSON under `data/splits/`. Noise injection (`src/data/label_noise.py`) operates on COCO dicts before YOLO conversion.

**COCO→YOLO conversion**: Both trainers convert COCO annotations to YOLO txt format at runtime, symlinking images into a YOLO directory layout. This happens in `_convert_coco_to_yolo()` methods on the trainer classes.

**Training**:
- `SupervisedTrainer` (src/training/supervised.py): Handles split creation, optional noise injection, COCO→YOLO conversion, then delegates to Ultralytics `model.train()`.
- `SemiSupervisedTrainer` (src/training/semi_supervised.py): Three-phase pipeline — supervised warmup → EMA teacher init → iterative pseudo-labelling rounds. Each round: teacher generates pseudo-labels → merge with labelled data → train student → update teacher EMA.

**EMA teacher** (src/models/ema.py): Shadow copy of student with warmup schedule on effective decay. Updates both parameters and buffers.

**Loss functions** (src/training/losses.py): Standard CE, Symmetric CE (Wang et al. ICCV 2019), Generalised CE (Zhang & Sabuncu NeurIPS 2018). Built via `build_loss_fn(cfg)`.

**Evaluation**: pycocotools-based mAP computation, per-class AP, and confusion matrices in `src/evaluation/metrics.py`. Visualisation utilities in `src/evaluation/visualise.py`.

## Key Design Decisions

- The `Detector` class (src/models/detector.py) wraps the Ultralytics YOLO Python API — all backbone interaction goes through this wrapper to make it swappable (e.g. RT-DETR).
- Category IDs in COCO are mapped to 0-indexed YOLO class indices via sorted category ID order.
- Splits are seeded and saved to JSON so experiments are reproducible across runs.
- Domain shift is simulated via Albumentations augmentation pipelines with severity levels (mild/moderate/severe) in `src/data/transforms.py`.
