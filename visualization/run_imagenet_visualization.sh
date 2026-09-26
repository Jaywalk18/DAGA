#!/bin/bash
# ============================================================================
# ImageNet Multi-Layer Attention Visualization Script
# 
# Visualizes how attention evolves across transformer layers for ImageNet
# classification, demonstrating DAGA's effectiveness on object recognition.
#
# Note: Even though the model was trained on 224x224 images, we can use
# higher resolution (518) for visualization to get finer-grained attention maps.
# DINOv2/v3 supports variable input sizes due to patch-based processing.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/results/imagenet_multilayer"

# Use only GPU 0 by default
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# Configuration
# ============================================================================

# ImageNet Dataset path
IMAGENET_PATH="/path/to/datasets/imagenet"

# Model checkpoint directories
CLASSIFICATION_BASELINE="$PROJECT_ROOT/outputs/classification/01_baseline"
CLASSIFICATION_DAGA="$PROJECT_ROOT/outputs/classification/05_daga_hourglass"

# Visualization parameters
NUM_SAMPLES=20
MAX_EVAL=1000  # Use more samples for accurate statistics across all classes

# Input size: Higher resolution gives finer attention maps
# Options: 224 (original), 518 (default DINOv2), 728 (high-res), 1024 (very high-res)
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
    echo "  input_size  - Input image size (default: 518)"
    echo "                224: Original training size"
    echo "                518: Default DINOv2 size (recommended)"
    echo "                768: High resolution for detailed attention"
    echo "  gpu_id      - GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0              # Use 518x518, GPU 0"
    echo "  $0 768          # Use 768x768 for higher resolution"
    echo "  $0 518 1        # Use GPU 1"
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

print_header "ImageNet Multi-Layer Attention Visualization"

echo "Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $IMAGENET_PATH"
echo "Output: $OUTPUT_BASE"
echo "Samples: $NUM_SAMPLES"
echo ""

# Find checkpoints
BASELINE_CKPT=$(find_latest_checkpoint "$CLASSIFICATION_BASELINE" "imagenet_baseline")
DAGA_CKPT=$(find_latest_checkpoint "$CLASSIFICATION_DAGA" "imagenet_daga")

if [ -z "$BASELINE_CKPT" ]; then
    echo "❌ Error: Baseline checkpoint not found!"
    echo "   Looking in: $CLASSIFICATION_BASELINE/imagenet_baseline*"
    exit 1
fi

if [ -z "$DAGA_CKPT" ]; then
    echo "❌ Error: DAGA checkpoint not found!"
    echo "   Looking in: $CLASSIFICATION_DAGA/imagenet_daga*"
    exit 1
fi

echo "✓ Baseline: $BASELINE_CKPT"
echo "✓ DAGA: $DAGA_CKPT"
echo ""

# Run visualization
python "$SCRIPT_DIR/visualize_imagenet_multilayer.py" \
    --baseline_checkpoint "$BASELINE_CKPT" \
    --daga_checkpoint "$DAGA_CKPT" \
    --data_path "$IMAGENET_PATH" \
    --output_dir "$OUTPUT_BASE" \
    --input_size "$INPUT_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --max_eval "$MAX_EVAL"

echo ""
print_header "Visualization Complete!"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Note on input size:"
echo "  - Training was done on 224x224"
echo "  - Visualization used ${INPUT_SIZE}x${INPUT_SIZE} for finer attention maps"
echo "  - DINOv2/v3 supports variable input sizes (patch-based processing)"
echo ""
echo "💡 Higher resolution inputs produce more detailed attention maps"
echo "   but require more GPU memory. Try 768 for even finer visualization."

