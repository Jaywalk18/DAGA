#!/bin/bash
# Logistic regression evaluation: Baseline vs DAGA on CIFAR-100
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/logreg"
GPU_IDS="${GPU_IDS:-0,1,2}"

DAGA_LAYERS="1 2 10 11"
BATCH_SIZE="${BATCH_SIZE:-256}"
INPUT_SIZE=224
NUM_WORKERS=6
MAX_ITER=1000
TOLERANCE=1e-12

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
    echo "  LogReg - $name"
    echo "========================================"
    
    local daga_args=""
    [ "$use_daga" = "true" ] && daga_args="--use_daga --daga_layers $DAGA_LAYERS"
    
    local pretrained="${CHECKPOINT_DIR}/${PRETRAINED_PATH}"
    [ -n "$checkpoint" ] && [ -f "$checkpoint" ] && pretrained="$checkpoint"
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        main_logreg.py \
        --dataset cifar100 \
        --data_path "/path/to/data/cifar" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "$pretrained" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS" \
        --max_iter "$MAX_ITER" \
        --tolerance "$TOLERANCE" \
        $daga_args \
        2>&1 | tee "$output_dir/eval.log"
    
    echo "✓ LogReg - $name done"
}

echo "========================================"
echo "  Logistic Regression on CIFAR-100"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "========================================"

run_exp "baseline" "false" ""
run_exp "daga" "true" ""

echo ""
echo "========================================"
echo "  Logistic regression done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
