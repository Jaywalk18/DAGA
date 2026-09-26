#!/bin/bash
# Common configuration for all training scripts
# This file should be sourced by run_classification.sh, run_detection.sh, run_segmentation.sh

# ============================================================================
# Signal Handling - Kill all child processes on exit
# ============================================================================
cleanup() {
    echo ""
    echo "Caught signal, terminating all processes..."
    kill -- -$$ 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM SIGHUP

# ============================================================================
# Environment Setup (modify according to your environment)
# ============================================================================
setup_environment() {
    # Activate conda environment (modify this path)
    # source /path/to/miniconda3/etc/profile.d/conda.sh
    # conda activate daga
    
    # DDP Environment Variables for multi-GPU training
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export OMP_NUM_THREADS=1
    export CUDA_LAUNCH_BLOCKING=0
}

# ============================================================================
# Common Configuration (modify these paths)
# ============================================================================
# Model settings
MODEL_NAME="${MODEL_NAME:-dinov3_vitb16}"
PRETRAINED_PATH="${PRETRAINED_PATH:-dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"

# Path settings (MODIFY THESE)
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname $(dirname $(realpath $0)))}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"

# GPU Configuration
DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-0,1}"
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"

# Training settings
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ============================================================================
# Helper Functions
# ============================================================================

get_num_gpus() {
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    echo ${#GPU_ARRAY[@]}
}

setup_paths() {
    cd "$PROJECT_ROOT"
    export PYTHONPATH=$PYTHONPATH:$(pwd)
}

print_config() {
    local task_name=$1
    local num_gpus=$(get_num_gpus)
    
    echo "DAGA ${task_name} Training"
    [[ -n "$SAMPLE_RATIO" ]] && echo "  Sample %:   ${SAMPLE_RATIO}"
    echo "==================================================================="
    echo "  Model:      ${MODEL_NAME}"
    echo "  GPU IDs:    ${GPU_IDS}"
    echo "  Num GPUs:   ${num_gpus}"
    echo "  Epochs:     ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE} per GPU (Total: $((BATCH_SIZE * num_gpus)))"
    echo "  LR:         ${LR}"
    [[ -n "$LAYERS_TO_USE" ]] && echo "  Layers:     ${LAYERS_TO_USE}"
    [[ -n "$OUT_INDICES" ]] && echo "  Out Layers: ${OUT_INDICES}"
    echo "==================================================================="
}

run_experiment() {
    local main_script=$1
    local exp_name=$2
    local description=$3
    shift 3
    
    local num_gpus=$(get_num_gpus)
    local output_subdir="${BASE_OUTPUT_DIR}/${exp_name}"
    mkdir -p "$output_subdir"
    
    echo -e "\n${description}"
    
    # Build sample_args based on script type
    local sample_args=()
    if [[ -n "$SAMPLE_RATIO" ]]; then
        if [[ "$main_script" == *"classification"* ]] || [[ "$main_script" == *"knn"* ]] || [[ "$main_script" == *"linear"* ]] || [[ "$main_script" == *"logreg"* ]]; then
            sample_args+=(--subset_ratio "$SAMPLE_RATIO")
        else
            sample_args+=(--sample_ratio "$SAMPLE_RATIO")
        fi
    fi
    
    # Build visualization args based on task type
    local vis_args=()
    if [[ "$main_script" == *"classification"* ]]; then
        vis_args+=(--vis_indices 1000 2000 3000 4000)
    elif [[ "$main_script" == *"detection"* ]] || [[ "$main_script" == *"segmentation"* ]] || [[ "$main_script" == *"depth"* ]]; then
        vis_args+=(--num_vis_samples "${NUM_VIS_SAMPLES:-4}")
    fi
    
    # Build task-specific args
    local task_args=()
    if [[ -n "$LAYERS_TO_USE" ]]; then
        task_args+=(--layers_to_use $LAYERS_TO_USE)
    fi
    if [[ -n "$OUT_INDICES" ]]; then
        task_args+=(--out_indices $OUT_INDICES)
    fi
    
    # Build training-specific args
    local training_args=()
    if [[ "$main_script" != *"linear"* ]] && [[ "$main_script" != *"knn"* ]] && [[ "$main_script" != *"logreg"* ]] && [[ "$main_script" != *"robustness"* ]] && [[ "$main_script" != *"retrieval"* ]]; then
        training_args+=(--epochs "$EPOCHS")
        training_args+=(--lr "$LR")
        training_args+=(--enable_visualization)
        training_args+=(--log_freq "${LOG_FREQ:-5}")
    elif [[ "$main_script" == *"retrieval"* ]]; then
        training_args+=(--epochs "$EPOCHS")
        training_args+=(--lr "$LR")
        training_args+=(--enable_visualization)
    fi
    
    # Handle absolute vs relative pretrained paths
    local pretrained_arg
    if [[ "$PRETRAINED_PATH" == /* ]]; then
        pretrained_arg="$PRETRAINED_PATH"
    else
        pretrained_arg="${CHECKPOINT_DIR}/${PRETRAINED_PATH}"
    fi
    
    # Use torchrun for DDP training
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$num_gpus \
        "$main_script" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "$pretrained_arg" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_subdir" \
        --num_workers "$NUM_WORKERS" \
        "${training_args[@]}" \
        "${sample_args[@]}" \
        "${vis_args[@]}" \
        "${task_args[@]}" \
        "$@"
    
    [ $? -eq 0 ] && echo "SUCCESS" || (echo "FAILED" && exit 1)
}
