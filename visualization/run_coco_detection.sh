#!/bin/bash
# ============================================================================
# COCO Detection Multi-Category Attention Visualization
#
# Features:
# - Multi-category attention with different colors per class
# - High resolution input (728 default, can use 1024)
# - Original patch-level attention display
# - Detection boxes overlaid with attention maps
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/results/coco_detection"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# Configuration
# ============================================================================

COCO_PATH="/path/to/datasets/COCO 2017"
DETECTION_BASELINE="$PROJECT_ROOT/outputs/detection/01_baseline"
DETECTION_DAGA="$PROJECT_ROOT/outputs/detection/03_daga_detection_four_layers"

NUM_SAMPLES=20
MAX_EVAL=2000

# Input size: Higher resolution = finer attention maps
# Options: 518 (standard), 728 (high), 1024 (very high)
INPUT_SIZE=728

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
    echo "Usage: $0 [input_size] [gpu_id]"
    echo ""
    echo "Options:"
    echo "  input_size  - Input image size (default: 728)"
    echo "                518: Standard resolution"
    echo "                728: High resolution (recommended)"
    echo "                1024: Very high resolution"
    echo "  gpu_id      - GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0              # Use 728x728, GPU 0"
    echo "  $0 1024         # Use 1024x1024 for finest attention"
    echo "  $0 728 1        # Use GPU 1"
}

cd "$PROJECT_ROOT"

# Parse arguments
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
    exit 0
fi

if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    INPUT_SIZE="$1"
fi

GPU_ID="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

print_header "COCO Detection Multi-Category Attention Visualization"

echo "Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $COCO_PATH"
echo "Output: $OUTPUT_BASE"
echo "Samples: $NUM_SAMPLES"
echo ""

# Find checkpoints
BASELINE_CKPT=$(find_latest_checkpoint "$DETECTION_BASELINE" "coco_baseline")
DAGA_CKPT=$(find_latest_checkpoint "$DETECTION_DAGA" "coco_daga")

if [ -z "$BASELINE_CKPT" ]; then
    echo "❌ Error: Baseline checkpoint not found!"
    echo "   Looking in: $DETECTION_BASELINE/coco_baseline*"
    exit 1
fi

if [ -z "$DAGA_CKPT" ]; then
    echo "❌ Error: DAGA checkpoint not found!"
    echo "   Looking in: $DETECTION_DAGA/coco_daga*"
    exit 1
fi

echo "✓ Baseline: $BASELINE_CKPT"
echo "✓ DAGA: $DAGA_CKPT"
echo ""

# Run visualization
python "$SCRIPT_DIR/visualize_coco_detection.py" \
    --baseline_checkpoint "$BASELINE_CKPT" \
    --daga_checkpoint "$DAGA_CKPT" \
    --data_path "$COCO_PATH" \
    --output_dir "$OUTPUT_BASE" \
    --input_size "$INPUT_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --max_eval "$MAX_EVAL"

print_header "Visualization Complete!"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── multiclass/     # Detection + multi-category attention"
echo "  │   └── XX_img_YYYYY_catsN.png"
echo "  └── attention_only/ # Pure attention comparison"
echo "      └── XX_img_YYYYY_attention.png"
echo ""
echo "💡 Features:"
echo "  - Different colors for different object categories"
echo "  - Patch-level attention (original resolution)"
echo "  - Upscaled attention for detail"
echo "  - Detection boxes with class labels"

