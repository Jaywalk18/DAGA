#!/bin/bash
# Depth estimation comparison: DAGA vs other PEFT methods on NYU Depth V2
# Methods: baseline, daga, lora, adapter-former, vit-adapter, layer-finetuning, full-finetune
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/depth"
GPU_IDS="${GPU_IDS:-0,1,2}"

# ============================================================
# Learning rates for each method (tuned for best performance)
# DAGA Best: LR=1e-3, Layers [2,5,8,11], RMSE=0.4310
# Layer-FT Best: RMSE=0.4135
# ============================================================
LR_BASELINE="${LR_BASELINE:-1e-3}"               # Frozen backbone
LR_FULL_FT="${LR_FULL_FT:-1e-4}"                 # Full fine-tuning
LR_PARTIAL_FT="${LR_PARTIAL_FT:-5e-4}"           # Layer fine-tuning
LR_LORA="${LR_LORA:-5e-4}"                       # LoRA
LR_ADAPTFORMER="${LR_ADAPTFORMER:-5e-4}"         # AdaptFormer
LR_VIT_ADAPTER="${LR_VIT_ADAPTER:-5e-4}"         # ViT-Adapter
LR_DAGA="${LR_DAGA:-1e-3}"                       # Best from tuning (RMSE=0.4310)

# Common settings
DAGA_LAYERS="2 5 8 11"
ADAPTER_LAYERS="2 5 8 11"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-12}"
INPUT_SIZE=518
NUM_WORKERS=6

# Batch sizes
BS_FROZEN="${BS_FROZEN:-16}"
BS_ADAPTER="${BS_ADAPTER:-12}"
BS_FINETUNE="${BS_FINETUNE:-8}"

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
    
    echo ""
    echo "========================================"
    echo "  Depth - $method"
    echo "  LR=$lr, BS=$batch_size"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_depth.py \
        --method "$method" \
        --dataset nyu_depth_v2 \
        --data_path "${DAGA_NYU_PATH:?Set DAGA_NYU_PATH to your NYU Depth V2 root}" \
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
        --adaptation_layers $ADAPTER_LAYERS \
        $extra_args \
        2>&1 | tee "$output_dir/train.log"
    
    echo "✓ Depth - $method done"
}

echo "========================================"
echo "  Depth Comparison (NYU Depth V2)"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Epochs: $EPOCHS"
echo "========================================"
echo ""
echo "Methods & Settings:"
echo "  1. Baseline:        LR=$LR_BASELINE"
echo "  2. DAGA (ours):     LR=$LR_DAGA"
echo "  3. LoRA:            LR=$LR_LORA"
echo "  4. Adapter-Former:  LR=$LR_ADAPTFORMER"
echo "  5. ViT-Adapter:     LR=$LR_VIT_ADAPTER"
echo "  6. Layer-FT:        LR=$LR_PARTIAL_FT"
echo "  7. Full-FT:         LR=$LR_FULL_FT"
echo ""

# ============================================================
# Run all methods
# ============================================================

# 1. Baseline (frozen backbone)
run_exp "baseline" "$LR_BASELINE" "$BS_FROZEN"

# 2. DAGA (ours)
run_exp "daga" "$LR_DAGA" "$BS_FROZEN"

# 3. LoRA
run_exp "lora" "$LR_LORA" "$BS_FROZEN"

# 4. Adapter-Former
run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTER"

# 5. ViT-Adapter
run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_ADAPTER"

# 6. Layer Fine-tuning
run_exp "layer-finetuning" "$LR_PARTIAL_FT" "$BS_FINETUNE" "--adaptation_layers 9 10 11"

# 7. Full Fine-tuning
run_exp "full-finetune" "$LR_FULL_FT" "$BS_FINETUNE"

echo ""
echo "========================================"
echo "  Depth experiments done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"

# Summary
echo ""
echo "=== Results Summary ==="
for method in baseline daga lora adapter-former vit-adapter layer-finetuning full-finetune; do
    log="${OUTPUT_BASE}/${method}/train.log"
    if [ -f "$log" ]; then
        best=$(grep "Best abs_rel" "$log" 2>/dev/null | tail -1)
        echo "$method: $best"
    fi
done
