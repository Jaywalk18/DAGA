#!/bin/bash
# DINOtxt Training Script
# Text-Image Alignment Training for DINOv3 with DAGA Support
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/dinotxt"
INPUT_SIZE=224
NUM_WORKERS=8
EPOCHS=10
LR=1e-4
BATCH_SIZE=64

# COCO Captions dataset
DATASET="coco_captions"
DATA_PATH="/path/to/coco"  # MODIFY THIS (should contain train2017, annotations)

# Optional: CLIP pretrained text encoder weights for initialization
CLIP_PATH=""  # MODIFY THIS if using CLIP initialization

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

# ============================================================================
# DINOtxt Training
# ============================================================================
echo "Running DINOtxt Training..."

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_dinotxt.py \
    --model_name "$MODEL_NAME" \
    --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
    --data_path "$DATA_PATH" \
    --input_size "$INPUT_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    --use_daga --daga_layers 1 2 10 11

echo "DINOtxt training completed!"
