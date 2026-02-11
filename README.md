# IBC-EDL

IBC + EDL for Long-tailed Classification with Uncertainty Estimation.

## Method

Combines:
- **IBC**: Instance-Based Calibration with multi-expert architecture and soft labels
- **EDL**: Evidential Deep Learning for uncertainty quantification

## Training Strategy

| Stage | Epochs | Loss |
|-------|--------|------|
| Warmup | 1-30 | CE + Balanced Softmax |
| IBC | 31-180 | IBC Soft Labels + Balanced Softmax |
| EDL | 181-200 | EDL (Digamma/Log/MSE) |

## Usage

```bash
python train.py \
    --dataset cifar10 \
    --imb_factor 0.01 \
    --loss edl_digamma \
    --warm_up 30 \
    --ibc_epochs 150 \
    --edl_epochs 20 \
    --gpuid 0
```

## Arguments

- `--dataset`: cifar10 or cifar100
- `--imb_factor`: Imbalance factor (0.01=extreme, 0.1=moderate)
- `--loss`: edl_digamma, edl_log, edl_mse
- `--k`: KNN neighbor ratio (default: 0.5)
- `--epsilon`: Soft label smoothing (default: 0.2)

## Files

- `train.py`: Main training script
- `models.py`: IBC-EDL model (PreActResNet18 + 3 heads)
- `losses.py`: EDL losses with soft label support
- `dataloader.py`: Long-tailed CIFAR dataloader
- `run_experiments.sh`: Experiment scripts
