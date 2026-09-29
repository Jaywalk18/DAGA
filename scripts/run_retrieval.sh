#!/bin/bash
# Instance Retrieval Evaluation Script
# Evaluates on ROxford5k and RParis6k datasets
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/retrieval"
INPUT_SIZE=518
NUM_WORKERS=4

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

# ============================================================================
# ROxford5k Retrieval
# ============================================================================
DATASET="roxford5k"
DATA_PATH="${DAGA_ROXFORD_PATH:?Set DAGA_ROXFORD_PATH to the ROxford5k root}"

echo "Running ROxford5k Retrieval..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python main_retrieval.py \
    --model_name "$MODEL_NAME" \
    --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_size "$INPUT_SIZE" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --use_daga --daga_layers 1 2 10 11 \
    --pooling gem

# ============================================================================
# RParis6k Retrieval
# ============================================================================
DATASET="rparis6k"
DATA_PATH="${DAGA_RPARIS_PATH:?Set DAGA_RPARIS_PATH to the RParis6k root}"

echo "Running RParis6k Retrieval..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python main_retrieval.py \
    --model_name "$MODEL_NAME" \
    --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --input_size "$INPUT_SIZE" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --use_daga --daga_layers 1 2 10 11 \
    --pooling gem

echo "Retrieval evaluation completed!"
