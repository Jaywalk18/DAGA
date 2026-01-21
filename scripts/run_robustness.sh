#!/bin/bash
# Robustness Evaluation Script
# Evaluates on ImageNet-C (corruptions)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/robustness"
INPUT_SIZE=224
NUM_WORKERS=8

# ImageNet-C dataset
DATASET="imagenet_c"
DATA_PATH="/path/to/imagenet-c"  # MODIFY THIS

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

# ============================================================================
# Robustness Evaluation
# ============================================================================
echo "Running Robustness Evaluation on ImageNet-C..."

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_robustness.py \
    --model_name "$MODEL_NAME" \
    --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_size "$INPUT_SIZE" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --use_daga --daga_layers 1 2 10 11

echo "Robustness evaluation completed!"
