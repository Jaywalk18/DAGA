#!/bin/bash
# Object Detection training script (COCO)
# DAGA with Hourglass configuration {1, 2, 10, 11}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/detection"
INPUT_SIZE=518
NUM_WORKERS=8
LOG_FREQ=100

# ============================================================================
# DAGA Configuration
# ============================================================================
run_daga_hourglass() {
    run_experiment "main_detection.py" "daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11 \
        --layers_to_use 4 8 11
}

# ============================================================================
# COCO Detection
# ============================================================================
DATASET="coco"
DATA_PATH="/path/to/coco"  # MODIFY THIS (should contain train2017, val2017, annotations)
EPOCHS=24
LR=1e-4
BATCH_SIZE=4

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Detection - COCO"
run_daga_hourglass
echo "Detection completed!"
