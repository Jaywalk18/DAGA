#!/bin/bash
# Logistic Regression Evaluation Script
# Uses official DINOv3 logreg evaluation with DAGA support
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/logreg"
INPUT_SIZE=224
NUM_WORKERS=8

# Dataset (ImageNet for logreg evaluation)
DATASET="imagenet"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the ImageNet-1K root}"

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

# ============================================================================
# Logistic Regression Evaluation
# ============================================================================
echo "Running Logistic Regression Evaluation..."

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_logreg.py \
    --model_name "$MODEL_NAME" \
    --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_size "$INPUT_SIZE" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --use_daga --daga_layers 1 2 10 11

echo "Logistic regression evaluation completed!"
