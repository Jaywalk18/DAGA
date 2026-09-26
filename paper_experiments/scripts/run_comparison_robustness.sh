#!/bin/bash
# Robustness evaluation on ImageNet-C: All Methods
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_config.sh"

OUTPUT_BASE="paper_experiments/outputs/robustness"
GPU_IDS="${GPU_IDS:-0,1,2}"

BATCH_SIZE="${BATCH_SIZE:-256}"
INPUT_SIZE=224
NUM_WORKERS=6

# ImageNet-C 数据路径
IMAGENET_C_PATH="/mnt/ssd/ImageNet-C/extracted"

# All 15 corruption types and 5 severity levels for full evaluation
CORRUPTION_TYPES="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
SEVERITY_LEVELS="1 2 3 4 5"

# Checkpoint 路径
CKPT_BASE="paper_experiments/outputs/classification/imagenet"

setup_environment
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p "$OUTPUT_BASE"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# 信号处理
trap 'echo "🛑 Caught signal, terminating..."; exit 1' INT TERM

run_robustness() {
    local method=$1
    local checkpoint=$2
    local layers=$3  # adaptation layers
    
    local output_dir="${OUTPUT_BASE}/${method}"
    mkdir -p "$output_dir"
    
    echo ""
    echo "========================================"
    echo "  Robustness - $method"
    echo "  Layers: $layers"
    echo "========================================"
    
    if [ -n "$checkpoint" ] && [ ! -f "$checkpoint" ]; then
        echo "⚠️ Checkpoint not found: $checkpoint"
        echo "Skipping $method"
        return 0
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        paper_experiments/main_comparison_robustness.py \
        --method "$method" \
        --adaptation_layers $layers \
        --data_path "$IMAGENET_C_PATH" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS" \
        --corruption_types $CORRUPTION_TYPES \
        --severity_levels $SEVERITY_LEVELS \
        --checkpoint "$checkpoint" \
        2>&1 | tee "$output_dir/eval.log"
    
    echo "✓ Robustness - $method done"
}

echo "========================================"
echo "  Robustness on ImageNet-C"
echo "  GPU: $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Data: $IMAGENET_C_PATH"
echo "========================================"

# Layer configurations (must match training checkpoints)
DAGA_LAYERS="1 2 10 11"       # DAGA used hourglass config
ADAPTER_LAYERS="2 5 8 11"      # Other adapters used spread config

# 运行所有方法
# 1. DAGA
DAGA_CKPT=$(find ${CKPT_BASE}/daga -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "daga" "$DAGA_CKPT" "$DAGA_LAYERS"

# 2. Baseline (no adaptation layers needed, but pass for consistency)
BASELINE_CKPT=$(find ${CKPT_BASE}/baseline -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "baseline" "$BASELINE_CKPT" "$ADAPTER_LAYERS"

# 3. Layer-Finetuning
LAYER_FT_CKPT=$(find ${CKPT_BASE}/layer-finetuning -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "layer-finetuning" "$LAYER_FT_CKPT" "$ADAPTER_LAYERS"

# 4. LoRA
LORA_CKPT=$(find ${CKPT_BASE}/lora -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "lora" "$LORA_CKPT" "$ADAPTER_LAYERS"

# 5. VPT-Deep
VPT_CKPT=$(find ${CKPT_BASE}/vpt-deep -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "vpt-deep" "$VPT_CKPT" "$ADAPTER_LAYERS"

# 6. Adapter-Former
ADAPTER_CKPT=$(find ${CKPT_BASE}/adapter-former -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "adapter-former" "$ADAPTER_CKPT" "$ADAPTER_LAYERS"

# 7. ViT-Adapter
VIT_ADAPTER_CKPT=$(find ${CKPT_BASE}/vit-adapter -name "best_model.pth" 2>/dev/null | head -1)
run_robustness "vit-adapter" "$VIT_ADAPTER_CKPT" "$ADAPTER_LAYERS"

# 8. Full-Finetune (可选 - 通常不做robustness测试)
# FULL_FT_CKPT=$(find ${CKPT_BASE}/full-finetune -name "best_model.pth" 2>/dev/null | head -1)
# run_robustness "full-finetune" "$FULL_FT_CKPT" "$ADAPTER_LAYERS"

echo ""
echo "========================================"
echo "  All robustness evaluations done!"
echo "  Results: $OUTPUT_BASE"
echo "========================================"
