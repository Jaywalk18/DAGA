#!/bin/bash
# ============================================================================
# Unified Visualization Script for All Tasks
# Generates attention maps and task-specific result visualizations
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/results"

# Use only GPU 0 by default (can be overridden with CUDA_VISIBLE_DEVICES env var)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================================
# Configuration - Update these paths to your trained models
# ============================================================================

# Classification models
CLASSIFICATION_BASELINE="$PROJECT_ROOT/outputs/classification/01_baseline"
CLASSIFICATION_DAGA="$PROJECT_ROOT/outputs/classification/04_daga_hourglass_layer"

# Segmentation models
SEGMENTATION_BASELINE="$PROJECT_ROOT/outputs/segmentation/01_baseline"
SEGMENTATION_DAGA="$PROJECT_ROOT/outputs/segmentation/02_daga_four_layers"

# Depth models
DEPTH_BASELINE="$PROJECT_ROOT/outputs/depth/01_baseline"
DEPTH_DAGA="$PROJECT_ROOT/outputs/depth/02_daga_feature_layers"

# Detection models
DETECTION_BASELINE="$PROJECT_ROOT/outputs/detection/01_baseline"
DETECTION_DAGA="$PROJECT_ROOT/outputs/detection/02_daga_detection_hourglass"

# Dataset paths
CIFAR_PATH="/path/to/datasets/cifar"
IMAGENET_PATH="/path/to/datasets/imagenet"
ADE20K_PATH="/path/to/datasets/ADE20K_2021_17_01"
NYU_PATH="/path/to/datasets/nyu_depth_v2_bts"
COCO_PATH="/path/to/datasets/COCO 2017"

NUM_SAMPLES=20

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
    
    # Find the latest experiment directory matching pattern
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
# Classification Visualization (Find Best Improvements)
# ============================================================================

visualize_classification() {
    print_header "Classification Visualization - Best Improvements"
    
    local datasets=("imagenet" "cifar100" "cifar10" "flowers102" "dtd" "pets" "cars" "food101" "sun397")
    local data_paths=(
        "$IMAGENET_PATH"
        "$CIFAR_PATH"
        "$CIFAR_PATH"
        "/path/to/datasets/OpenDataLab___Oxford_102_Flower/raw"
        "/path/to/datasets/OpenDataLab___DTD/raw/dtd"
        "/path/to/datasets/OpenDataLab___Oxford-IIIT_Pets/raw"
        "/path/to/datasets/OpenDataLab___Stanford_Cars/raw/Stanford_Cars"
        "/path/to/datasets/food-101"
        "/path/to/datasets/OpenDataLab___SUN397/raw/SUN397"
    )
    
    for i in "${!datasets[@]}"; do
        local dataset="${datasets[$i]}"
        local data_path="${data_paths[$i]}"
        
        echo ""
        echo "📊 Dataset: $dataset"
        
        # Find both checkpoints
        local baseline_ckpt=$(find_latest_checkpoint "$CLASSIFICATION_BASELINE" "${dataset}_baseline")
        local daga_ckpt=$(find_latest_checkpoint "$CLASSIFICATION_DAGA" "${dataset}_daga")
        
        if [ -n "$baseline_ckpt" ] && [ -n "$daga_ckpt" ]; then
            echo "  ✓ Baseline: $baseline_ckpt"
            echo "  ✓ DAGA: $daga_ckpt"
            
            # Compare mode: generate both best_accuracy and best_improvement
            python "$SCRIPT_DIR/visualize_all_tasks.py" \
                --task classification \
                --mode compare \
                --selection_mode both \
                --baseline_checkpoint "$baseline_ckpt" \
                --daga_checkpoint "$daga_ckpt" \
                --data_path "$data_path" \
                --output_dir "$OUTPUT_BASE/classification/${dataset}" \
                --num_samples "$NUM_SAMPLES" \
                --max_eval 1000
        else
            echo "  ⚠️ Missing checkpoint(s) for $dataset"
            [ -z "$baseline_ckpt" ] && echo "    - Baseline not found"
            [ -z "$daga_ckpt" ] && echo "    - DAGA not found"
        fi
    done
}

# ============================================================================
# Segmentation Visualization (Find Best Improvements)
# ============================================================================

visualize_segmentation() {
    print_header "Segmentation Visualization"
    
    local baseline_ckpt=$(find_latest_checkpoint "$SEGMENTATION_BASELINE" "ade20k_baseline")
    local daga_ckpt=$(find_latest_checkpoint "$SEGMENTATION_DAGA" "ade20k_daga")
    
    if [ -n "$baseline_ckpt" ] && [ -n "$daga_ckpt" ]; then
        echo "✓ Baseline: $baseline_ckpt"
        echo "✓ DAGA: $daga_ckpt"
        
        python "$SCRIPT_DIR/visualize_all_tasks.py" \
            --task segmentation \
            --mode compare \
            --selection_mode both \
            --baseline_checkpoint "$baseline_ckpt" \
            --daga_checkpoint "$daga_ckpt" \
            --data_path "$ADE20K_PATH" \
            --output_dir "$OUTPUT_BASE/segmentation/ade20k" \
            --num_samples "$NUM_SAMPLES" \
            --max_eval 500
    else
        echo "⚠️ Missing checkpoint(s) for ADE20K"
        [ -z "$baseline_ckpt" ] && echo "  - Baseline not found"
        [ -z "$daga_ckpt" ] && echo "  - DAGA not found"
    fi
}

# ============================================================================
# Depth Visualization (Find Best Improvements)
# ============================================================================

visualize_depth() {
    print_header "Depth Estimation Visualization"
    
    local baseline_ckpt=$(find_latest_checkpoint "$DEPTH_BASELINE" "nyu_depth_v2_baseline")
    local daga_ckpt=$(find_latest_checkpoint "$DEPTH_DAGA" "nyu_depth_v2_daga")
    
    if [ -n "$baseline_ckpt" ] && [ -n "$daga_ckpt" ]; then
        echo "✓ Baseline: $baseline_ckpt"
        echo "✓ DAGA: $daga_ckpt"
        
        python "$SCRIPT_DIR/visualize_all_tasks.py" \
            --task depth \
            --mode compare \
            --selection_mode both \
            --baseline_checkpoint "$baseline_ckpt" \
            --daga_checkpoint "$daga_ckpt" \
            --data_path "$NYU_PATH" \
            --output_dir "$OUTPUT_BASE/depth/nyu" \
            --num_samples "$NUM_SAMPLES" \
            --max_eval 500
    else
        echo "⚠️ Missing checkpoint(s) for NYU Depth"
        [ -z "$baseline_ckpt" ] && echo "  - Baseline not found"
        [ -z "$daga_ckpt" ] && echo "  - DAGA not found"
    fi
}

# ============================================================================
# Detection Visualization (Find Best Improvements)
# ============================================================================

visualize_detection() {
    print_header "Detection Visualization"
    
    local baseline_ckpt=$(find_latest_checkpoint "$DETECTION_BASELINE" "coco_baseline")
    local daga_ckpt=$(find_latest_checkpoint "$DETECTION_DAGA" "coco_daga")
    
    if [ -n "$baseline_ckpt" ] && [ -n "$daga_ckpt" ]; then
        echo "✓ Baseline: $baseline_ckpt"
        echo "✓ DAGA: $daga_ckpt"
        
        python "$SCRIPT_DIR/visualize_all_tasks.py" \
            --task detection \
            --mode compare \
            --selection_mode both \
            --baseline_checkpoint "$baseline_ckpt" \
            --daga_checkpoint "$daga_ckpt" \
            --data_path "$COCO_PATH" \
            --output_dir "$OUTPUT_BASE/detection/coco" \
            --num_samples "$NUM_SAMPLES" \
            --max_eval 500
    else
        echo "⚠️ Missing checkpoint(s) for COCO Detection"
        [ -z "$baseline_ckpt" ] && echo "  - Baseline not found"
        [ -z "$daga_ckpt" ] && echo "  - DAGA not found"
    fi
}

# ============================================================================
# Main
# ============================================================================

usage() {
    echo "Usage: $0 [task] [gpu_id]"
    echo ""
    echo "Tasks:"
    echo "  classification  - Visualize classification models"
    echo "  segmentation    - Visualize segmentation models"
    echo "  depth           - Visualize depth estimation models"
    echo "  detection       - Visualize detection models"
    echo "  all             - Visualize all tasks (default)"
    echo ""
    echo "GPU:"
    echo "  gpu_id          - GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 classification      # Use GPU 0"
    echo "  $0 classification 1    # Use GPU 1"
    echo "  $0 all 0               # All tasks on GPU 0"
    echo "  CUDA_VISIBLE_DEVICES=2 $0 all  # Use GPU 2"
}

cd "$PROJECT_ROOT"

TASK="${1:-all}"
GPU_ID="${2:-0}"

# Override GPU if second argument provided
if [ -n "$2" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "============================================================================"
echo "🎨 DAGA Visualization Tool"
echo "============================================================================"
echo "Task: $TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_BASE"
echo "Samples per model: $NUM_SAMPLES"
echo "============================================================================"

case "$TASK" in
    classification)
        visualize_classification
        ;;
    segmentation)
        visualize_segmentation
        ;;
    depth)
        visualize_depth
        ;;
    detection)
        visualize_detection
        ;;
    all)
        visualize_classification
        visualize_segmentation
        visualize_depth
        visualize_detection
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown task: $TASK"
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "✅ Visualization Complete!"
echo "============================================================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── classification/"
echo "  │   ├── cifar100_improvements/  (Best DAGA improvements)"
echo "  │   └── ..."
echo "  ├── segmentation/"
echo "  │   └── ade20k_improvements/"
echo "  ├── depth/"
echo "  │   └── nyu_improvements/"
echo "  └── detection/"
echo "      ├── coco_daga/"
echo "      └── coco_baseline/"
echo ""
echo "💡 Each *_improvements folder contains samples where DAGA"
echo "   shows the largest improvement over the baseline model."

