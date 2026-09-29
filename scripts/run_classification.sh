#!/bin/bash
# Classification training script
# DAGA with Hourglass configuration {1, 2, 10, 11}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/classification"
INPUT_SIZE=224
NUM_WORKERS=6
SAMPLE_RATIO=""
LOG_FREQ=5

# ============================================================================
# DAGA Hourglass Configuration {1, 2, 10, 11}
# ============================================================================
run_daga_hourglass() {
    run_experiment "main_classification.py" "daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11
}

# ============================================================================
# Dataset Configuration (MODIFY THESE PATHS)
# ============================================================================
# Example: CIFAR-100
DATASET="cifar100"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the dataset root}"
EPOCHS=30
LR=0.5
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - ${DATASET}"
run_daga_hourglass
echo "Classification completed!"

# ============================================================================
# Additional datasets can be added below with their configs:
#
# CIFAR-10:      EPOCHS=30, LR=0.5, BATCH_SIZE=256
# CIFAR-100:     EPOCHS=30, LR=0.5, BATCH_SIZE=256
# Flowers-102:   EPOCHS=50, LR=0.1, BATCH_SIZE=32
# DTD:           EPOCHS=50, LR=0.5, BATCH_SIZE=64
# Oxford Pets:   EPOCHS=50, LR=0.1, BATCH_SIZE=256
# Stanford Cars: EPOCHS=50, LR=0.2, BATCH_SIZE=256
# Food-101:      EPOCHS=20, LR=0.2, BATCH_SIZE=256
# SUN397:        EPOCHS=30, LR=0.2, BATCH_SIZE=256
# ImageNet:      EPOCHS=30, LR=0.3, BATCH_SIZE=256
# ============================================================================
