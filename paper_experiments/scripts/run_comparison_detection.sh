#!/bin/bash
# Detection comparison: DAGA vs other PEFT methods on COCO
# Methods: Full FT, Partial FT, LoRA, AdaptFormer, ViT-Adapter, DAGA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/detection"
GPU_IDS="${GPU_IDS:-0,1,2}"

# ============================================================
# Learning rates for each method (based on Optuna + literature)
# Detection Optuna: Best LR=0.007, hourglass {1,2,10,11}, mAP=0.41
# ============================================================
LR_BASELINE="${LR_BASELINE:-1e-3}"               # Frozen backbone, train head
LR_FULL_FT="${LR_FULL_FT:-1e-4}"                 # Full fine-tuning
LR_PARTIAL_FT="${LR_PARTIAL_FT:-5e-4}"           # Partial fine-tuning
LR_LORA="${LR_LORA:-1e-3}"                       # LoRA
LR_ADAPTFORMER="${LR_ADAPTFORMER:-2e-3}"         # AdaptFormer
LR_VIT_ADAPTER="${LR_VIT_ADAPTER:-1e-3}"         # ViT-Adapter
LR_VPT="${LR_VPT:-1e-3}"                         # VPT-Deep
LR_DAGA="${LR_DAGA:-0.007}"                      # Optimized from Optuna

# Common settings - DAGA uses hourglass config from Optuna
DAGA_LAYERS="1 2 10 11"
ADAPTER_LAYERS="2 5 8 11"

# Training settings (matching Optuna search config)
EPOCHS="${EPOCHS:-50}"
INPUT_SIZE=518
NUM_WORKERS=8

# Batch size per GPU - matched with Optuna search (A100 80GB)
# Optuna used BS=128 for detection
BS_FROZEN="${BS_FROZEN:-128}"       # Frozen backbone methods (DAGA, baseline, LoRA, VPT)
BS_ADAPTER="${BS_ADAPTER:-96}"      # Adapter methods (slightly more memory)
BS_FINETUNE="${BS_FINETUNE:-48}"    # Finetuning methods (gradients for backbone)

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
    echo "  Detection - $method"
    echo "  LR=$lr, BS=$batch_size x $NUM_GPUS = $effective_bs"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_detection.py \
        --method "$method" \
        --dataset coco \
        --data_path "/mnt/ssd/coco" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --batch_size "$batch_size" \
        --input_size "$INPUT_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$lr" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS" \
        --use_amp \
        --adaptation_layers $ADAPTER_LAYERS \
        $extra_args \
        2>&1 | tee "$output_dir/train.log"
    
    echo "✓ Detection - $method done"
}

echo "========================================"
echo "  Detection Comparison (COCO)"
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
echo "  Layer-FT:      $LR_PARTIAL_FT / $BS_FINETUNE"
echo "  DAGA (ours):   $LR_DAGA / $BS_FROZEN (Optuna best)"
echo ""

# ============================================================
# Run all methods in order of expected training time
# ============================================================

# 1. DAGA (ours) - main method, frozen backbone (BS=128 from Optuna)
run_exp "daga" "$LR_DAGA" "$BS_FROZEN" "--adaptation_layers $DAGA_LAYERS --layers_to_use $DAGA_LAYERS"

# 2. Baseline (frozen backbone, train head only)
run_exp "baseline" "$LR_BASELINE" "$BS_FROZEN"

# 3. Other PEFT methods (frozen backbone)
run_exp "lora" "$LR_LORA" "$BS_FROZEN"
run_exp "vpt-deep" "$LR_VPT" "$BS_FROZEN"

# 4. Adapter methods (slightly more memory)
run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTER"
run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_ADAPTER"

# 5. Fine-tuning methods (slower, more memory)
run_exp "layer-finetuning" "$LR_PARTIAL_FT" "$BS_FINETUNE"

# 6. Full fine-tuning (slowest, needs smaller batch due to full gradients)
BS_FULL_FT="${BS_FULL_FT:-24}"
run_exp "full-finetune" "$LR_FULL_FT" "$BS_FULL_FT"

echo ""
echo "========================================"
echo "  Detection experiments done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
