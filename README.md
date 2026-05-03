# Semi-Supervised Object Detection for Waste Stream Analysis

Training robust object detectors when most of your data is unlabelled and the labels you *do* have are noisy — the reality of deploying computer vision in recycling facilities.

## Motivation

Waste sorting facilities generate enormous volumes of visual data from conveyor-belt cameras, but only a tiny fraction gets annotated. Labels that do exist are often noisy: objects are crushed, occluded, and visually ambiguous. Models also need to generalise across facilities with different lighting, backgrounds, and waste compositions.

This project tackles all three challenges on the [ZeroWaste](https://zenodo.org/records/6412647) dataset:

1. **Semi-supervised learning** — leveraging large pools of unlabelled waste imagery via teacher-student pseudo-labelling
2. **Noisy label robustness** — training reliably when annotations contain class-flip errors, bounding box jitter, and missing objects
3. **Domain shift** — evaluating how well models transfer across different visual environments

## Key Results

*Coming soon — experiments in progress.*

## Method

### Teacher-Student Pseudo-Labelling

A supervised "teacher" model (YOLOv8) is trained on a small labelled subset, then generates pseudo-labels on the unlabelled pool. A "student" model trains on both real and pseudo-labelled data. The teacher is updated as an exponential moving average (EMA) of the student, creating a self-improving loop.

### Noisy Label Strategies

We inject controlled noise (class flips, bbox jitter, missing annotations) into the labelled set and compare standard cross-entropy against noise-robust alternatives (symmetric CE, generalised CE).

### Domain Shift Evaluation

We evaluate cross-scene generalisation and robustness to deployment-time visual shifts (lighting, contrast, blur).

## Setup

```bash
# Clone
git clone [https://github.com/Smithy305/zerowaste-semi-supervised.git](https://github.com/Smithy305/waste-stream.git)
cd zerowaste-semi-supervised

# Install dependencies
pip install -r requirements.txt

# Download ZeroWaste dataset
bash data/download_zerowaste.sh
```

## Usage

```bash
# Supervised baseline (full labels)
python scripts/train_supervised.py --config configs/supervised_baseline.yaml

# Supervised baseline (10% labels)
python scripts/train_supervised.py --config configs/supervised_baseline.yaml labelled_fraction=0.1

# Semi-supervised teacher-student
python scripts/train_semi_supervised.py --config configs/semi_supervised.yaml labelled_fraction=0.1

# Evaluate
python scripts/evaluate.py --config configs/semi_supervised.yaml --checkpoint results/best.pt

# Run all ablations
bash scripts/run_ablations.sh
```

## Experiments

| Experiment | What it tests |
|---|---|
| Labelled fraction sweep (5–100%) | How much does semi-supervised learning help at different annotation budgets? |
| Pseudo-label threshold ablation | Trading precision vs recall in pseudo-labels |
| EMA decay ablation | How aggressively should the teacher track the student? |
| Noise rate sweep | Robustness of different loss functions to label corruption |
| Domain shift evaluation | Cross-scene generalisation |

## Project Structure

```
├── configs/           # YAML experiment configs
├── data/              # Dataset download scripts and docs
├── src/
│   ├── data/          # Dataset loading, augmentation, noise injection, splits
│   ├── models/        # Detector wrapper, EMA
│   ├── training/      # Supervised and semi-supervised training loops
│   ├── evaluation/    # Metrics and visualisation
│   └── utils/         # Config, logging, seeding
├── scripts/           # Training and evaluation entry points
├── notebooks/         # Data exploration and results analysis
└── results/           # Checkpoints and figures (gitignored)
```

## Tech Stack

PyTorch · Ultralytics YOLOv8 · Albumentations · OmegaConf · Weights & Biases · pycocotools

## Dataset

[ZeroWaste](https://zenodo.org/records/6412647) — instance-level annotations of waste objects in cluttered recycling scenes (COCO format). Categories include rigid plastic, cardboard, metal, soft plastic, and more.

## References

- Bashkirova et al., "ZeroWaste: Towards Deformable Object Segmentation in Cluttered Scenes," CVPR 2023
- Liu et al., "Unbiased Teacher for Semi-Supervised Object Detection," ICLR 2021
- Xu et al., "Soft Teacher: End-to-End Semi-Supervised Object Detection," ICCV 2021
- Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy Labels," ICCV 2019

## Author

Joe Smith — [joe-smith-computer-vision.com](https://joe-smith-computer-vision.com)

## License

MIT
