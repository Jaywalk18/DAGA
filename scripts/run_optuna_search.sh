#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============ Search Settings ============
N_TRIALS="${N_TRIALS:-12}"
EPOCHS="${EPOCHS:-5}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/optuna_search}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-0}"

# ============ Batch Sizes (per GPU) ============
export BS_CLASSIFICATION="${BS_CLASSIFICATION:-1024}"
export BS_DETECTION="${BS_DETECTION:-128}"
export BS_SEGMENTATION="${BS_SEGMENTATION:-48}"
export BS_DEPTH="${BS_DEPTH:-12}"

# ============ LR Ranges ============
export LR_MIN_CLASSIFICATION="${LR_MIN_CLASSIFICATION:-0.1}"
export LR_MAX_CLASSIFICATION="${LR_MAX_CLASSIFICATION:-1.0}"
export LR_MIN_DETECTION="${LR_MIN_DETECTION:-0.0001}"
export LR_MAX_DETECTION="${LR_MAX_DETECTION:-0.1}"
export LR_MIN_SEGMENTATION="${LR_MIN_SEGMENTATION:-0.0001}"
export LR_MAX_SEGMENTATION="${LR_MAX_SEGMENTATION:-0.1}"
export LR_MIN_DEPTH="${LR_MIN_DEPTH:-0.0001}"
export LR_MAX_DEPTH="${LR_MAX_DEPTH:-0.1}"

# ============ Subset Ratios (per task, overrides global SUBSET_RATIO) ============
export SUBSET_CLASSIFICATION="${SUBSET_CLASSIFICATION:-$SUBSET_RATIO}"
export SUBSET_DETECTION="${SUBSET_DETECTION:-$SUBSET_RATIO}"
export SUBSET_SEGMENTATION="${SUBSET_SEGMENTATION:-$SUBSET_RATIO}"
export SUBSET_DEPTH="${SUBSET_DEPTH:-$SUBSET_RATIO}"

setup_environment
cd "$PROJECT_ROOT"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export GPU_IDS="${GPU_IDS:-0,1,2}"

SWANLAB_FLAG=""
[ "$ENABLE_SWANLAB" = "1" ] && SWANLAB_FLAG="--enable_swanlab"

run_task() {
    TASK=$1
    BS_VAR="BS_${TASK^^}"
    LR_MIN_VAR="LR_MIN_${TASK^^}"
    LR_MAX_VAR="LR_MAX_${TASK^^}"
    SUBSET_VAR="SUBSET_${TASK^^}"
    
    echo ""
    echo "========================================"
    echo "  $TASK"
    echo "========================================"
    echo "  Trials=$N_TRIALS Epochs=$EPOCHS"
    echo "  BS=${!BS_VAR} LR=[${!LR_MIN_VAR}, ${!LR_MAX_VAR}] Subset=${!SUBSET_VAR}"
    echo "========================================"
    
    python scripts/optuna_search.py \
        --task "$TASK" \
        --n_trials "$N_TRIALS" \
        --epochs "$EPOCHS" \
        --subset_ratio "$SUBSET_RATIO" \
        --output_dir "$OUTPUT_DIR" \
        $SWANLAB_FLAG
    
    echo "✓ $TASK done"
}

echo "Optuna Hyperparameter Search"
echo ""

# run_task "classification"  # Done: LR=0.9, hourglass
# run_task "detection"  # Done: LR=0.007, hourglass, mAP=0.41
# run_task "segmentation"  # Done: LR=0.00046, spread, 45.89%
run_task "depth"

echo ""
echo "All done! Results: $OUTPUT_DIR/"
