#!/bin/bash
# Classification comparison: DAGA vs other PEFT methods
# Methods: Linear Probe, Full FT, Partial FT, VPT, LoRA, AdaptFormer, ViT-Adapter, DAGA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/classification"
GPU_IDS="${GPU_IDS:-0,1,2}"

# ============================================================
# Learning rates for each method (based on literature/empirical)
# ============================================================
LR_LINEAR_PROBE="${LR_LINEAR_PROBE:-0.01}"       # Frozen backbone, train head only
LR_FULL_FT="${LR_FULL_FT:-1e-4}"                 # Small LR to preserve pretrained weights
LR_PARTIAL_FT="${LR_PARTIAL_FT:-5e-4}"           # Fine-tune last 3 blocks
LR_VPT="${LR_VPT:-0.025}"                        # VPT prompts need moderate LR
LR_LORA="${LR_LORA:-1e-3}"                       # LoRA typical range
LR_ADAPTFORMER="${LR_ADAPTFORMER:-5e-3}"         # AdaptFormer adapters
LR_VIT_ADAPTER="${LR_VIT_ADAPTER:-1e-3}"         # ViT-Adapter
LR_DAGA="${LR_DAGA:-0.9}"                        # Optimized from Optuna

# ============================================================
# Batch sizes for each method (maximize GPU utilization)
# ============================================================
BS_LINEAR_PROBE="${BS_LINEAR_PROBE:-1024}"       # Frozen backbone, max batch
BS_FULL_FT="${BS_FULL_FT:-64}"                   # Full FT needs gradients for all params
BS_PARTIAL_FT="${BS_PARTIAL_FT:-256}"            # Partial FT, some gradients
BS_VPT="${BS_VPT:-1024}"                         # VPT frozen backbone
BS_LORA="${BS_LORA:-1024}"                       # LoRA frozen backbone
BS_ADAPTFORMER="${BS_ADAPTFORMER:-1024}"         # AdaptFormer frozen backbone
BS_VIT_ADAPTER="${BS_VIT_ADAPTER:-512}"          # ViT-Adapter slightly more memory
BS_DAGA="${BS_DAGA:-1024}"                       # DAGA optimized (frozen backbone)

# Common settings
DAGA_LAYERS="1 2 10 11"
ADAPTER_LAYERS="2 5 8 11"

EPOCHS="${EPOCHS:-20}"
INPUT_SIZE="${INPUT_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-6}"

# SwanLab settings
ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"

setup_environment
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p "$OUTPUT_BASE"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Run classification experiment
run_exp() {
    local method=$1
    local lr=$2
    local batch_size=$3
    local dataset=$4
    local data_path=$5
    local extra_args="${6:-}"
    
    local output_dir="${OUTPUT_BASE}/${dataset}/${method}"
    mkdir -p "$output_dir"
    
    local effective_bs=$((batch_size * NUM_GPUS))
    
    echo ""
    echo "========================================"
    echo "  $dataset - $method"
    echo "  LR=$lr, BS=$batch_size x $NUM_GPUS = $effective_bs"
    echo "========================================"
    
    local swanlab_args=""
    if [ "$ENABLE_SWANLAB" = "1" ]; then
        swanlab_args="--enable_swanlab --swanlab_name ${dataset}_${method}"
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_classification.py \
        --method "$method" \
        --dataset "$dataset" \
        --data_path "$data_path" \
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
        $swanlab_args \
        $extra_args \
        2>&1 | tee "$output_dir/train.log"
    
    echo "✓ $dataset - $method done"
}

# Run all methods on a dataset
run_all_methods() {
    local dataset=$1
    local data_path=$2
    
    echo ""
    echo "========================================================"
    echo "  Dataset: $dataset"
    echo "========================================================"
    
    # === COMPLETED ===
    # run_exp "baseline" "$LR_LINEAR_PROBE" "$BS_LINEAR_PROBE" "$dataset" "$data_path"
    # run_exp "layer-finetuning" "$LR_PARTIAL_FT" "$BS_PARTIAL_FT" "$dataset" "$data_path"
    # run_exp "lora" "$LR_LORA" "$BS_LORA" "$dataset" "$data_path"
    # run_exp "adapter-former" "$LR_ADAPTFORMER" "$BS_ADAPTFORMER" "$dataset" "$data_path"
    # run_exp "vpt-deep" "$LR_VPT" "$BS_VPT" "$dataset" "$data_path"
    # run_exp "vit-adapter" "$LR_VIT_ADAPTER" "$BS_VIT_ADAPTER" "$dataset" "$data_path"
    
    # === TO RUN (DAGA optimized - no extra forward pass) ===
    run_exp "daga" "$LR_DAGA" "$BS_DAGA" "$dataset" "$data_path" "--adaptation_layers $DAGA_LAYERS"
    run_exp "full-finetune" "$LR_FULL_FT" "$BS_FULL_FT" "$dataset" "$data_path"
}

echo "========================================"
echo "  Classification Comparison Experiments"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Epochs: $EPOCHS"
echo "  SwanLab: $([ "$ENABLE_SWANLAB" = "1" ] && echo "ON (Dino_DAGA)" || echo "OFF")"
echo "========================================"
echo ""
echo "Method settings (LR / BatchSize per GPU):"
echo "  Linear Probe:  $LR_LINEAR_PROBE / $BS_LINEAR_PROBE"
echo "  Full FT:       $LR_FULL_FT / $BS_FULL_FT"
echo "  Partial FT:    $LR_PARTIAL_FT / $BS_PARTIAL_FT"
echo "  VPT:           $LR_VPT / $BS_VPT"
echo "  LoRA:          $LR_LORA / $BS_LORA"
echo "  AdaptFormer:   $LR_ADAPTFORMER / $BS_ADAPTFORMER"
echo "  ViT-Adapter:   $LR_VIT_ADAPTER / $BS_VIT_ADAPTER"
echo "  DAGA (ours):   $LR_DAGA / $BS_DAGA"
echo ""

# ============================================================
# Main experiments
# ============================================================

# ImageNet-1K (main benchmark)
run_all_methods "imagenet" "${DAGA_IMAGENET_PATH:?Set DAGA_IMAGENET_PATH to your ImageNet-1K root}"

# Fine-grained datasets (uncomment as needed)
# run_all_methods "cifar100" "/path/to/data/cifar"
# run_all_methods "flowers102" "/path/to/data/flowers102"
# run_all_methods "dtd" "/path/to/data/dtd"
# run_all_methods "pets" "/path/to/data/pets"
# run_all_methods "cars" "/path/to/data/cars"
# run_all_methods "food101" "/path/to/data/food101"
# run_all_methods "sun397" "/path/to/data/sun397"

echo ""
echo "========================================"
echo "  All classification experiments done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
