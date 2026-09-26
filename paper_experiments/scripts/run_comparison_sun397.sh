#!/bin/bash
# SUN397 Scene Classification: DAGA vs other PEFT methods
# 8 Methods: Baseline, Layer-FT, Full-FT, LoRA, VPT-Deep, AdaptFormer, ViT-Adapter, DAGA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/classification/sun397"
DATA_PATH="${DAGA_SUN397_PATH:?Set DAGA_SUN397_PATH to your SUN397 root}"
GPU_IDS="${GPU_IDS:-0,1,2}"

# Learning rates (same as ImageNet)
LR_BASELINE="0.01"
LR_LAYER_FT="5e-4"
LR_FULL_FT="1e-4"
LR_LORA="1e-3"
LR_VPT="0.025"
LR_ADAPTFORMER="5e-3"
LR_VIT_ADAPTER="1e-3"
LR_DAGA="0.9"

# Batch sizes
BS_BASELINE="1024"
BS_LAYER_FT="256"
BS_FULL_FT="64"
BS_LORA="1024"
BS_VPT="1024"
BS_ADAPTFORMER="1024"
BS_VIT_ADAPTER="512"
BS_DAGA="1024"

# Common settings
DAGA_LAYERS="1 2 10 11"
ADAPTER_LAYERS="2 5 8 11"
EPOCHS="${EPOCHS:-20}"
INPUT_SIZE="224"
NUM_WORKERS="6"

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
    
    # Skip if already completed
    if [ -f "$output_dir/results.json" ]; then
        echo "⏭️  $method already completed, skipping..."
        return 0
    fi
    
    local effective_bs=$((batch_size * NUM_GPUS))
    
    echo ""
    echo "========================================"
    echo "  SUN397 - $method"
    echo "  LR=$lr, BS=$batch_size x $NUM_GPUS = $effective_bs"
    echo "  Time: $(date)"
    echo "========================================"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_classification.py \
        --method "$method" \
        --dataset "sun397" \
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
        --adaptation_layers $ADAPTER_LAYERS \
        $extra_args \
        2>&1 | tee "$output_dir/train.log"
    
    echo "✓ SUN397 - $method done at $(date)"
}

echo "========================================"
echo "  SUN397 Scene Classification"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Epochs: $EPOCHS"
echo "  Data: $DATA_PATH"
echo "========================================"

# Run all 8 methods
run_exp "baseline" "$LR_BASELINE" "$BS_BASELINE"
run_exp "layer-finetuning" "$LR_LAYER_FT" "$BS_LAYER_FT"
run_exp "full-finetune" "$LR_FULL_FT" "$BS_FULL_FT"
run_exp "lora" "$LR_LORA" "$BS_LORA"
run_exp "vpt-deep" "$LR_VPT" "$BS_VPT"
run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTFORMER"
run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_VIT_ADAPTER"
run_exp "daga" "$LR_DAGA" "$BS_DAGA" "--adaptation_layers $DAGA_LAYERS"

echo ""
echo "========================================"
echo "  SUN397 All methods done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for method in baseline layer-finetuning full-finetune lora vpt-deep adapter-former vit-adapter daga; do
    result_file="$OUTPUT_BASE/$method/results.json"
    if [ -f "$result_file" ]; then
        acc=$(python3 -c "import json; print(f'{json.load(open(\"$result_file\"))[\"best_acc\"]:.2f}%')" 2>/dev/null || echo "N/A")
        echo "  $method: $acc"
    fi
done

