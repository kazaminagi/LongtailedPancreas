#!/bin/bash
# IBC-EDL Experiments
# Training: Warmup (30) -> IBC (150) -> EDL (20)

cd "$(dirname "$0")"

# ============================================
# Basic Experiments
# ============================================

# CIFAR-10, EDL Digamma (recommended)
python train.py \
    --dataset cifar10 \
    --imb_factor 0.01 \
    --loss edl_digamma \
    --warm_up 30 \
    --ibc_epochs 150 \
    --edl_epochs 20 \
    --gpuid 0 \
    --name ibc_edl_digamma_imb0.01

# ============================================
# Compare EDL Losses
# ============================================

for loss in edl_digamma edl_log edl_mse; do
    python train.py \
        --dataset cifar10 \
        --imb_factor 0.01 \
        --loss $loss \
        --warm_up 30 \
        --ibc_epochs 150 \
        --edl_epochs 20 \
        --gpuid 0 \
        --name ibc_edl_${loss}_imb0.01
done

# ============================================
# Compare Imbalance Factors
# ============================================

for imb in 0.01 0.02 0.1; do
    python train.py \
        --dataset cifar10 \
        --imb_factor $imb \
        --loss edl_digamma \
        --warm_up 30 \
        --ibc_epochs 150 \
        --edl_epochs 20 \
        --gpuid 0 \
        --name ibc_edl_digamma_imb${imb}
done

# ============================================
# Ablation: K values
# ============================================

for k in 0.3 0.5 0.7; do
    python train.py \
        --dataset cifar10 \
        --imb_factor 0.01 \
        --loss edl_digamma \
        --k $k \
        --warm_up 30 \
        --ibc_epochs 150 \
        --edl_epochs 20 \
        --gpuid 0 \
        --name ibc_edl_digamma_k${k}
done

# ============================================
# Ablation: Epsilon values
# ============================================

for eps in 0.1 0.2 0.3; do
    python train.py \
        --dataset cifar10 \
        --imb_factor 0.01 \
        --loss edl_digamma \
        --epsilon $eps \
        --warm_up 30 \
        --ibc_epochs 150 \
        --edl_epochs 20 \
        --gpuid 0 \
        --name ibc_edl_digamma_eps${eps}
done

# ============================================
# Ablation: EDL epochs
# ============================================

for edl_ep in 10 20 30 50; do
    python train.py \
        --dataset cifar10 \
        --imb_factor 0.01 \
        --loss edl_digamma \
        --warm_up 30 \
        --ibc_epochs 150 \
        --edl_epochs $edl_ep \
        --gpuid 0 \
        --name ibc_edl_digamma_edlep${edl_ep}
done

# ============================================
# CIFAR-100
# ============================================

python train.py \
    --dataset cifar100 \
    --imb_factor 0.01 \
    --loss edl_digamma \
    --warm_up 30 \
    --ibc_epochs 150 \
    --edl_epochs 20 \
    --gpuid 0 \
    --name ibc_edl_digamma_cifar100_imb0.01

echo "All experiments completed!"
