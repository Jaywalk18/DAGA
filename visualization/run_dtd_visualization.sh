#!/bin/bash
# ============================================================================
# DTD Multi-Layer Attention Visualization Script
# 
# Visualizes how attention evolves across transformer layers for texture
# classification, demonstrating DAGA's effectiveness on mid-level features.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/results/dtd_multilayer"

# Use only GPU 0 by default
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# Configuration
# ============================================================================

# DTD Dataset path
DTD_PATH="/path/to/datasets/OpenDataLab___DTD/raw/dtd"

# Model checkpoint directories
CLASSIFICATION_BASELINE="$PROJECT_ROOT/outputs/classification/01_baseline"
CLASSIFICATION_DAGA="$PROJECT_ROOT/outputs/classification/05_daga_hourglass"

# Visualization parameters
NUM_SAMPLES=20
MAX_EVAL=500
INPUT_SIZE=518

# ============================================================================
# Helper Functions
# ============================================================================

find_latest_checkpoint() {
    local base_dir="$1"
    local pattern="$2"
    
    if [ ! -d "$base_dir" ]; then
        echo ""
        return
    fi
    
    local latest_dir=$(find "$base_dir" -maxdepth 1 -type d -name "${pattern}*" | sort -r | head -1)
    
    if [ -n "$latest_dir" ] && [ -f "$latest_dir/best_model.pth" ]; then
        echo "$latest_dir/best_model.pth"
    else
        echo ""
    fi
}

print_header() {
    echo ""
    echo "============================================================================"
    echo "🎨 $1"
    echo "============================================================================"
}

# ============================================================================
# Main
# ============================================================================

usage() {
    echo "Usage: $0 [gpu_id]"
    echo ""
    echo "Options:"
    echo "  gpu_id    - GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0        # Use GPU 0"
    echo "  $0 1      # Use GPU 1"
    echo "  CUDA_VISIBLE_DEVICES=2 $0  # Use GPU 2"
}

cd "$PROJECT_ROOT"

GPU_ID="${1:-0}"

# Override GPU if argument provided
if [ -n "$1" ] && [ "$1" != "-h" ] && [ "$1" != "--help" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

print_header "DTD Multi-Layer Attention Visualization"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DTD_PATH"
echo "Output: $OUTPUT_BASE"
echo "Samples: $NUM_SAMPLES"
echo ""

# Find checkpoints
BASELINE_CKPT=$(find_latest_checkpoint "$CLASSIFICATION_BASELINE" "dtd_baseline")
DAGA_CKPT=$(find_latest_checkpoint "$CLASSIFICATION_DAGA" "dtd_daga")

if [ -z "$BASELINE_CKPT" ]; then
    echo "❌ Error: Baseline checkpoint not found!"
    echo "   Looking in: $CLASSIFICATION_BASELINE/dtd_baseline*"
    exit 1
fi

if [ -z "$DAGA_CKPT" ]; then
    echo "❌ Error: DAGA checkpoint not found!"
    echo "   Looking in: $CLASSIFICATION_DAGA/dtd_daga*"
    exit 1
fi

echo "✓ Baseline: $BASELINE_CKPT"
echo "✓ DAGA: $DAGA_CKPT"
echo ""

# Run visualization
python "$SCRIPT_DIR/visualize_dtd_multilayer.py" \
    --baseline_checkpoint "$BASELINE_CKPT" \
    --daga_checkpoint "$DAGA_CKPT" \
    --data_path "$DTD_PATH" \
    --output_dir "$OUTPUT_BASE" \
    --input_size "$INPUT_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --max_eval "$MAX_EVAL"

echo ""
print_header "Visualization Complete!"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── best_improvement/     # Side-by-side Baseline vs DAGA comparison"
echo "  │   └── multilayer_*.png  # Multi-layer attention visualization"
echo "  └── attention_evolution/  # Single-model attention progression"
echo "      ├── baseline_*.png"
echo "      └── daga_*.png"
echo ""
echo "💡 These visualizations show how attention evolves across layers,"
echo "   demonstrating DAGA's ability to focus on discriminative texture units."

