#!/bin/bash
# Segmentation comparison: DAGA vs other PEFT methods on ADE20K
# Methods: baseline, daga, vit-adapter, adapter-former, lora, vpt-deep, layer-finetuning, full-finetune
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/segmentation"
GPU_IDS="${GPU_IDS:-0,1,2}"

# ============================================================
# Learning rates for each method (tuned for best performance)
# DAGA Best: LR=3e-4, Layers [8,9,10,11], mIoU=49.25%
# Layer-FT Best: LR=1e-3, Layers [9,10,11], mIoU=50.12%
# ============================================================
LR_BASELINE="${LR_BASELINE:-1e-3}"               # Frozen backbone, train head
LR_FULL_FT="${LR_FULL_FT:-1e-4}"                 # Full fine-tuning
LR_PARTIAL_FT="${LR_PARTIAL_FT:-1e-3}"           # Layer fine-tuning (tuned)
LR_LORA="${LR_LORA:-5e-4}"                       # LoRA
LR_ADAPTFORMER="${LR_ADAPTFORMER:-1e-3}"         # AdaptFormer
LR_VIT_ADAPTER="${LR_VIT_ADAPTER:-5e-4}"         # ViT-Adapter
LR_VPT="${LR_VPT:-1e-3}"                         # VPT-Deep
LR_DAGA="${LR_DAGA:-3e-4}"                       # Best from tuning (49.25% mIoU)

# Common settings - DAGA uses deep layers for best performance
DAGA_LAYERS="8 9 10 11"                          # Best: 49.25% mIoU
ADAPTER_LAYERS="2 5 8 11"

# Training settings
EPOCHS="${EPOCHS:-50}"
INPUT_SIZE=518
NUM_WORKERS=6

# Batch size per GPU (A100 80GB)
BS_FROZEN="${BS_FROZEN:-48}"       # Frozen backbone methods
BS_ADAPTER="${BS_ADAPTER:-32}"     # Adapter methods
BS_FINETUNE="${BS_FINETUNE:-16}"   # Finetuning methods

setup_environment
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p "$OUTPUT_BASE"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

run_exp() {
    local method=$1
    local lr=$2
    local batch_size=$3
    local extra_args="${4:-}"
    
    local output_dir="${OUTPUT_BASE}/${method}"
    mkdir -p "$output_dir"
    
    local effective_bs=$((batch_size * NUM_GPUS))
    
    echo ""
    echo "========================================"
    echo "  Segmentation - $method"
    echo "  LR=$lr, BS=$batch_size x $NUM_GPUS = $effective_bs"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_segmentation.py \
        --method "$method" \
        --dataset ade20k \
        --data_path "${DAGA_ADE20K_PATH:?Set DAGA_ADE20K_PATH to your ADE20K root}" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --batch_size "$batch_size" \
        --input_size "$INPUT_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$lr" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS" \
        --use_amp \
        --out_indices $ADAPTER_LAYERS \
        $extra_args \
        2>&1 | tee "$output_dir/train.log"
    
    echo "✓ Segmentation - $method done"
}

echo "========================================"
echo "  Segmentation Comparison (ADE20K)"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Epochs: $EPOCHS"
echo "========================================"
echo ""
echo "Settings (LR / BatchSize per GPU):"
echo "  Baseline:      $LR_BASELINE / $BS_FROZEN"
echo "  LoRA:          $LR_LORA / $BS_FROZEN"
echo "  VPT-Deep:      $LR_VPT / $BS_FROZEN"
echo "  AdaptFormer:   $LR_ADAPTFORMER / $BS_ADAPTER"
echo "  ViT-Adapter:   $LR_VIT_ADAPTER / $BS_ADAPTER"
echo "  Layer-FT:      $LR_PARTIAL_FT / $BS_FINETUNE (Best: 50.12%)"
echo "  Full-FT:       $LR_FULL_FT / $BS_FINETUNE"
echo "  DAGA (ours):   $LR_DAGA / $BS_FROZEN [8,9,10,11] (Best: 49.25%)"
echo ""

# ============================================================
# Run all methods in order of expected training time
# ============================================================

# 1. DAGA (ours) - main method, frozen backbone
run_exp "daga" "$LR_DAGA" "$BS_FROZEN" "--adaptation_layers $DAGA_LAYERS"

# 2. Baseline (frozen backbone, train head only)
run_exp "baseline" "$LR_BASELINE" "$BS_FROZEN"

# 3. Other PEFT methods (frozen backbone)
run_exp "lora" "$LR_LORA" "$BS_FROZEN" "--adaptation_layers $ADAPTER_LAYERS"
run_exp "vpt-deep" "$LR_VPT" "$BS_FROZEN"

# 4. Adapter methods (slightly more memory)
run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTER" "--adaptation_layers $ADAPTER_LAYERS"
run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_ADAPTER" "--adaptation_layers $ADAPTER_LAYERS"

# 5. Fine-tuning methods (slower, more memory)
run_exp "layer-finetuning" "$LR_PARTIAL_FT" "$BS_FINETUNE" "--adaptation_layers 9 10 11"

# 6. Full fine-tuning (slowest)
run_exp "full-finetune" "$LR_FULL_FT" "$BS_FINETUNE"

echo ""
echo "========================================"
echo "  Segmentation experiments done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
