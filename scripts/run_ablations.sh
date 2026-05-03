#!/usr/bin/env bash
# Run all ablation experiments defined in the project spec.
# Usage: bash scripts/run_ablations.sh
set -euo pipefail

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo " ZeroWaste Semi-Supervised Ablations"
echo "========================================"

# --- 1. Labelled fraction sweep (supervised baseline) ---
echo ""
echo "=== Labelled Fraction Sweep (Supervised) ==="
for frac in 0.05 0.1 0.2 0.5 1.0; do
    echo "--- labelled_fraction=$frac ---"
    python scripts/train_supervised.py \
        --config configs/supervised_baseline.yaml \
        labelled_fraction=$frac \
        experiment_name="supervised_frac${frac}"
done

# --- 2. Labelled fraction sweep (semi-supervised) ---
echo ""
echo "=== Labelled Fraction Sweep (Semi-Supervised) ==="
for frac in 0.05 0.1 0.2 0.5; do
    echo "--- labelled_fraction=$frac ---"
    python scripts/train_semi_supervised.py \
        --config configs/semi_supervised.yaml \
        labelled_fraction=$frac \
        experiment_name="semi_sup_frac${frac}"
done

# --- 3. Pseudo-label threshold ablation ---
echo ""
echo "=== Pseudo-Label Threshold Ablation ==="
for thresh in 0.5 0.6 0.7 0.8 0.9; do
    echo "--- pl_threshold=$thresh ---"
    python scripts/train_semi_supervised.py \
        --config configs/semi_supervised.yaml \
        semi_supervised.pl_threshold=$thresh \
        experiment_name="pl_thresh_${thresh}"
done

# --- 4. EMA decay ablation ---
echo ""
echo "=== EMA Decay Ablation ==="
for decay in 0.99 0.999 0.9999; do
    echo "--- ema_decay=$decay ---"
    python scripts/train_semi_supervised.py \
        --config configs/semi_supervised.yaml \
        semi_supervised.ema_decay=$decay \
        experiment_name="ema_decay_${decay}"
done

# --- 5. Noise rate sweep (class flip) ---
echo ""
echo "=== Noise Rate Sweep (Class Flip) ==="
for rate in 0.0 0.1 0.2 0.3; do
    echo "--- noise_rate=$rate, loss=cross_entropy ---"
    python scripts/train_supervised.py \
        --config configs/noisy_labels.yaml \
        noise.noise_rate=$rate \
        noise.noise_type=class_flip \
        training.loss_fn=cross_entropy \
        experiment_name="noise_cf_${rate}_ce"

    echo "--- noise_rate=$rate, loss=symmetric_ce ---"
    python scripts/train_supervised.py \
        --config configs/noisy_labels.yaml \
        noise.noise_rate=$rate \
        noise.noise_type=class_flip \
        training.loss_fn=symmetric_ce \
        experiment_name="noise_cf_${rate}_sce"

    echo "--- noise_rate=$rate, loss=generalised_ce ---"
    python scripts/train_supervised.py \
        --config configs/noisy_labels.yaml \
        noise.noise_rate=$rate \
        noise.noise_type=class_flip \
        training.loss_fn=generalised_ce \
        experiment_name="noise_cf_${rate}_gce"
done

# --- 6. Noise type comparison ---
echo ""
echo "=== Noise Type Comparison ==="
for noise_type in class_flip bbox_jitter missing_annot; do
    echo "--- noise_type=$noise_type ---"
    python scripts/train_supervised.py \
        --config configs/noisy_labels.yaml \
        noise.noise_type=$noise_type \
        noise.noise_rate=0.2 \
        experiment_name="noise_type_${noise_type}"
done

echo ""
echo "========================================"
echo " All ablations complete!"
echo "========================================"
echo " Run 'python scripts/generate_figures.py' to produce plots."
