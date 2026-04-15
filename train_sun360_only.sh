#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Hyperparameters — edit here
# ---------------------------------------------------------------------------
DATA_DIR="data/sun360_torchfeed"
DEVICE="cuda:2"
EPOCHS=2000
PRETRAIN_EPOCHS=50      # fixed in config, shown here for reference
BATCH_SIZE=32
LR="1e-3"
WEIGHT_DECAY="5e-3"
LAMBDA_POLICY="1.0"
T=6

# ---------------------------------------------------------------------------
# Auto-construct wandb run name from hparams
# ---------------------------------------------------------------------------
RUN_NAME="sun360only_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}_T${T}_lp${LAMBDA_POLICY}"

echo "========================================="
echo "  Run name : ${RUN_NAME}"
echo "  Data     : ${DATA_DIR}"
echo "  Device   : ${DEVICE}"
echo "  Epochs   : phase1=${PRETRAIN_EPOCHS}  phase2=${EPOCHS}"
echo "  Batch    : ${BATCH_SIZE}  LR: ${LR}  WD: ${WEIGHT_DECAY}"
echo "  T        : ${T}  lambda_policy: ${LAMBDA_POLICY}"
echo "========================================="

uv run python train.py \
    --data-dir      "${DATA_DIR}" \
    --device        "${DEVICE}" \
    --epochs        "${EPOCHS}" \
    --batch-size    "${BATCH_SIZE}" \
    --run-name      "${RUN_NAME}" \
    --wandb
