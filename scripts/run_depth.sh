#!/bin/bash
# Depth Estimation training script
# DAGA with Hourglass configuration {1, 2, 10, 11}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/depth"
INPUT_SIZE=518
NUM_WORKERS=8
LOG_FREQ=100

# ============================================================================
# DAGA Configuration
# ============================================================================
run_daga_hourglass() {
    run_experiment "main_depth.py" "daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11
}

# ============================================================================
# Dataset Configuration (MODIFY THESE PATHS)
# ============================================================================
# NYU Depth V2
DATASET="nyu"
DATA_PATH="/path/to/nyu_depth_v2"  # MODIFY THIS
EPOCHS=25
LR=1e-4
BATCH_SIZE=8

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Depth - ${DATASET}"
run_daga_hourglass
echo "Depth estimation completed!"

# ============================================================================
# Additional datasets:
#
# NYU Depth V2: EPOCHS=25, LR=1e-4, BATCH_SIZE=8
# KITTI:        EPOCHS=25, LR=5e-5, BATCH_SIZE=8
# ============================================================================
