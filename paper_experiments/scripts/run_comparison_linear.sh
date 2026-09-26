#!/bin/bash
# Linear probing evaluation: Baseline vs DAGA on ImageNet
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/linear"
GPU_IDS="${GPU_IDS:-0,1,2}"

DAGA_LAYERS="1 2 10 11"
BATCH_SIZE="${BATCH_SIZE:-128}"
INPUT_SIZE=224
NUM_WORKERS=6
LINEAR_EPOCHS=1
EPOCH_LENGTH=2000
LEARNING_RATES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2"

setup_environment
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p "$OUTPUT_BASE"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

run_exp() {
    local name=$1
    local use_daga=$2
    local checkpoint=$3
    
    local output_dir="${OUTPUT_BASE}/${name}"
    mkdir -p "$output_dir"
    
    echo ""
    echo "========================================"
    echo "  Linear Probe - $name"
    echo "========================================"
    
    local daga_args=""
    [ "$use_daga" = "true" ] && daga_args="--use_daga --daga_layers $DAGA_LAYERS"
    
    local pretrained="${CHECKPOINT_DIR}/${PRETRAINED_PATH}"
    [ -n "$checkpoint" ] && [ -f "$checkpoint" ] && pretrained="$checkpoint"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        main_linear.py \
        --dataset imagenet \
        --data_path "${DAGA_IMAGENET_PATH:?Set DAGA_IMAGENET_PATH to your ImageNet-1K root}" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "$pretrained" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS" \
        --linear_epochs "$LINEAR_EPOCHS" \
        --epoch_length "$EPOCH_LENGTH" \
        --learning_rates $LEARNING_RATES \
        $daga_args \
        2>&1 | tee "$output_dir/eval.log"
    
    echo "✓ Linear Probe - $name done"
}

echo "========================================"
echo "  Linear Probing on ImageNet"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "========================================"

run_exp "baseline" "false" ""
run_exp "daga" "true" ""

echo ""
echo "========================================"
echo "  Linear probing done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
