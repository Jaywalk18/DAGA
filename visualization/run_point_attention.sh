#!/bin/bash
# ============================================================================
# Category-based Attention Visualization
#
# Shows attention distribution from detected object centers.
# Each attention map corresponds to a detected object category (person, car, etc.)
# Similar to DINO/DINOv2 attention visualization style.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/results/category_attention"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"  # Use GPU 1 (40GB)

# Configuration
COCO_PATH="/path/to/datasets/COCO 2017"

# Use pretrained models directly (not detection checkpoints)
# Baseline: ViT-S pretrained (0.08 GB, 22M params)
BASELINE_CKPT="$PROJECT_ROOT/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
# DAGA: ViT-7B pretrained (25 GB, 7B params) - requires 40GB GPU
DAGA_CKPT="$PROJECT_ROOT/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

# Use pretrained mode (not detection checkpoint mode)
USE_PRETRAINED_BASELINE=true
USE_PRETRAINED_DAGA=true

NUM_SAMPLES=30
MAX_EVAL=5000  # Evaluate all images in val2017
INPUT_SIZE=1024  # Higher resolution for clearer visualization
SAVE_SEPARATE=true  # Save to separate folders with individual images and metrics

cd "$PROJECT_ROOT"

echo ""
echo "============================================================================"
echo "🎨 Category-based Attention Visualization"
echo "============================================================================"

if [ ! -f "$BASELINE_CKPT" ]; then
    echo "❌ Baseline checkpoint not found: $BASELINE_CKPT"
    exit 1
fi

if [ ! -f "$DAGA_CKPT" ]; then
    echo "❌ DAGA checkpoint not found: $DAGA_CKPT"
    exit 1
fi

echo "✓ Baseline (ViT-S): $BASELINE_CKPT"
echo "✓ DAGA (ViT-7B): $DAGA_CKPT"
echo "✓ GPU: $CUDA_VISIBLE_DEVICES"
echo "✓ Input size: $INPUT_SIZE"

# Run visualization
echo "Running visualization..."
python "$SCRIPT_DIR/visualize_point_attention.py" \
    --baseline_checkpoint "$BASELINE_CKPT" \
    --daga_checkpoint "$DAGA_CKPT" \
    --data_path "$COCO_PATH" \
    --output_dir "$OUTPUT_BASE" \
    --input_size "$INPUT_SIZE" \
    --num_samples "$NUM_SAMPLES" \
    --max_eval "$MAX_EVAL" \
    $([ "$USE_PRETRAINED_BASELINE" = true ] && echo "--use_pretrained_baseline") \
    $([ "$USE_PRETRAINED_DAGA" = true ] && echo "--use_pretrained_daga") \
    $([ "$SAVE_SEPARATE" = true ] && echo "--save_separate")

echo ""
echo "✅ Done! Results: $OUTPUT_BASE"
echo ""

if [ "$SAVE_SEPARATE" = true ]; then
    echo "Generated folder structure (--save_separate mode):"
    echo "  $OUTPUT_BASE/"
    echo "  ├── 01_img_XXXXX/"
    echo "  │   ├── daga/"
    echo "  │   │   ├── 00_original.png     - Original image with bounding boxes"
    echo "  │   │   ├── 01_person.png       - Similarity map for each category"
    echo "  │   │   ├── 02_car.png"
    echo "  │   │   ├── ..."
    echo "  │   │   └── metrics.txt         - Quality metrics for DAGA model"
    echo "  │   ├── baseline/"
    echo "  │   │   ├── 00_original.png"
    echo "  │   │   ├── 01_person.png"
    echo "  │   │   ├── ..."
    echo "  │   │   └── metrics.txt         - Quality metrics for baseline"
    echo "  │   └── comparison_summary.txt  - Side-by-side comparison"
    echo "  ├── 02_img_YYYYY/"
    echo "  │   └── ..."
    echo "  └── ..."
else
    echo "Generated files (combined mode):"
    echo "  - XX_img_YYYYY_daga.png: DAGA attention per detected object (3x3 grid)"
    echo "  - XX_img_YYYYY_baseline.png: Baseline attention per object (3x3 grid)"
fi
echo ""
echo "Models used:"
echo "  - Baseline: ViT-S (dinov3_vits16, 22M params)"
echo "  - DAGA: ViT-7B (dinov3_vit7b16, 7B params)"
echo ""
echo "Each similarity map shows feature correlation from the object center."
echo "Labels show the object category (person, car, dog, etc.)"

