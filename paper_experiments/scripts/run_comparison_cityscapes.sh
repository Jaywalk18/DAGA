#!/bin/bash
# Cityscapes Segmentation comparison: DAGA vs other PEFT methods
# Methods: baseline, daga, vit-adapter, adapter-former, lora, layer-finetuning, full-finetune
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dinov3_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/cityscapes"
GPU_IDS="${GPU_IDS:-0,1,2}"
DATA_PATH="/datasets/OpenDataLab___CityScapes/raw"

# ============================================================
# Learning rates for each method (based on ADE20K tuning)
# ============================================================
LR_BASELINE="${LR_BASELINE:-1e-3}"
LR_FULL_FT="${LR_FULL_FT:-1e-4}"
LR_PARTIAL_FT="${LR_PARTIAL_FT:-1e-3}"
LR_LORA="${LR_LORA:-5e-4}"
LR_ADAPTFORMER="${LR_ADAPTFORMER:-1e-3}"
LR_VIT_ADAPTER="${LR_VIT_ADAPTER:-5e-4}"
LR_DAGA="${LR_DAGA:-3e-4}"

# DAGA uses deep layers for segmentation
DAGA_LAYERS="8 9 10 11"
ADAPTER_LAYERS="2 5 8 11"

# Training settings
EPOCHS="${EPOCHS:-50}"
INPUT_SIZE=518
NUM_WORKERS=8

# Batch size per GPU (A100 80GB)
BS_FROZEN="${BS_FROZEN:-48}"
BS_ADAPTER="${BS_ADAPTER:-32}"
BS_FINETUNE="${BS_FINETUNE:-16}"

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
    echo "  Cityscapes Segmentation - $method"
    echo "  LR=$lr, BS=$batch_size x $NUM_GPUS = $effective_bs"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_segmentation.py \
        --method "$method" \
        --dataset cityscapes \
        --data_path "$DATA_PATH" \
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
    
    echo ">>> $method completed <<<"
}

echo "========================================"
echo "  Cityscapes Segmentation Experiments"
echo "  Dataset: Cityscapes (19 classes)"
echo "  Train: 2,975 images, Val: 500 images"
echo "========================================"

# Run all methods
run_exp "baseline" "$LR_BASELINE" "$BS_FROZEN"
run_exp "daga" "$LR_DAGA" "$BS_ADAPTER" "--adaptation_layers $DAGA_LAYERS"
run_exp "layer-finetuning" "$LR_PARTIAL_FT" "$BS_FINETUNE" "--finetune_layers 9 10 11"
run_exp "lora" "$LR_LORA" "$BS_ADAPTER"
run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTER"
run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_ADAPTER"
run_exp "full-finetune" "$LR_FULL_FT" "$BS_FINETUNE"

echo ""
echo "========================================"
echo "  All Cityscapes experiments completed!"
echo "========================================"

# Summary
echo ""
echo "=== Results Summary ==="
for method in baseline daga layer-finetuning lora adapter-former vit-adapter full-finetune; do
    LOG="${OUTPUT_BASE}/${method}/train.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep -i "best\|Best" "$LOG" | tail -1 || echo "N/A")
        echo "$method: $BEST"
    fi
done

conda deactivate

