#!/bin/bash
# Semantic Segmentation training script
# DAGA with Hourglass configuration {1, 2, 10, 11}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/segmentation"
INPUT_SIZE=518
NUM_WORKERS=8
LOG_FREQ=50

# ============================================================================
# DAGA Configuration
# ============================================================================
run_daga_hourglass() {
    run_experiment "main_segmentation.py" "daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11 \
        --out_indices 4 8 11
}

# ============================================================================
# Dataset Configuration (MODIFY THESE PATHS)
# ============================================================================
# ADE20K (150 classes)
DATASET="ade20k"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the ADE20K root}"
EPOCHS=40
LR=1e-4
BATCH_SIZE=8

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Segmentation - ${DATASET}"
run_daga_hourglass
echo "Segmentation completed!"

# ============================================================================
# Additional datasets:
#
# ADE20K:      EPOCHS=40, LR=1e-4, BATCH_SIZE=8
# VOC2012:     EPOCHS=80, LR=5e-5, BATCH_SIZE=8
# Cityscapes:  EPOCHS=40, LR=5e-5, BATCH_SIZE=4
# ============================================================================
