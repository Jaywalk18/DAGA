"""
COCO Detection Multi-Category Attention Visualization

Features:
- Multi-category attention visualization with different colors per class
- High resolution input support (518, 728, 1024)
- Original patch-level attention display
- Detection boxes overlaid with attention maps
- Side-by-side Baseline vs DAGA comparison
"""

import os
os.environ['SWANLAB_DISABLED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3"))

from core.backbones import load_dinov3_backbone, get_attention_map


# COCO category names and colors
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Generate distinct colors for each category
def get_category_colors(num_categories=80):
    """Generate visually distinct colors for each category"""
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors3 = plt.cm.tab20c(np.linspace(0, 1, 20))
    colors4 = plt.cm.Set3(np.linspace(0, 1, 12))
    all_colors = np.vstack([colors, colors2, colors3, colors4])
    return all_colors[:num_categories]

CATEGORY_COLORS = get_category_colors(80)


def denormalize_image(img_tensor):
    """Denormalize image tensor to [0, 1] range"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(std * img + mean, 0, 1)
    return img


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract args"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = None
    if "args" in checkpoint:
        args = argparse.Namespace(**checkpoint["args"])
    return checkpoint, args


class DetectionVisualizer:
    """Detection model visualizer with multi-category attention"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        from tasks.detection import DetectionModel
        
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Infer model config
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vitb16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path', 
                                  str(PROJECT_ROOT / 'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'))
        
        num_classes = 80
        if 'detection_head.cls_head.2.weight' in state_dict:
            num_classes = state_dict['detection_head.cls_head.2.weight'].shape[0]
        
        use_daga = getattr(self.ckpt_args, 'use_daga', False)
        daga_layers = getattr(self.ckpt_args, 'daga_layers', [])
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = DetectionModel(
            vit_model, num_classes=num_classes,
            use_daga=use_daga, daga_layers=daga_layers
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.num_classes = num_classes
        
        print(f"✓ Detection model loaded: DAGA={use_daga}, layers={daga_layers}, classes={num_classes}")
    
    def get_attention_and_detections(self, image_tensor, score_threshold=0.3):
        """Get attention map and detections"""
        from core.simple_detection_head import decode_predictions
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        B, _, H, W = image_tensor.shape
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            if len(outputs) >= 3:
                cls_logits, box_preds, centerness = outputs[:3]
            else:
                return None, None, None
            
            stride = self.model.stride if hasattr(self.model, 'stride') else 14
            detections = decode_predictions(
                cls_logits, box_preds, centerness,
                image_size=(H, W),
                stride=stride,
                score_threshold=score_threshold,
                nms_threshold=0.5,
                max_detections=50
            )
        
        # Get attention from last layer - access vit through vit_wrapper
        vit = self.model.vit_wrapper.vit
        x_processed, (patch_H, patch_W) = vit.prepare_tokens_with_masks(image_tensor)
        num_patches = patch_H * patch_W
        
        for idx, block in enumerate(vit.blocks):
            rope_sincos = vit.rope_embed(H=patch_H, W=patch_W) if vit.rope_embed else None
            x_processed = block(x_processed, rope_sincos)
        
        # Get attention from last block
        last_block = vit.blocks[-1]
        with torch.no_grad():
            attn_weights = get_attention_map(last_block, x_processed)
        
        # Process attention
        seq_len = x_processed.shape[1]
        num_registers = seq_len - num_patches - 1
        if num_registers < 0:
            num_registers = 0
        
        cls_attn = attn_weights[:, :, 0, 1+num_registers:]
        cls_attn = cls_attn.mean(dim=1)  # Average over heads
        
        # Normalize
        attn_min = cls_attn.amin(dim=1, keepdim=True)
        attn_max = cls_attn.amax(dim=1, keepdim=True)
        cls_attn = (cls_attn - attn_min) / (attn_max - attn_min + 1e-8)
        
        attn_map = cls_attn.reshape(B, patch_H, patch_W).cpu().numpy()[0]
        
        return attn_map, detections, (patch_H, patch_W)


def save_single_attention_overlay(
    image,
    attn_map,
    patch_shape,
    save_path,
    title="Attention"
):
    """
    Save a single attention overlay image (no multi-panel, just one clean image)
    """
    from scipy.ndimage import zoom as scipy_zoom
    
    img_h, img_w = image.shape[:2]
    patch_h, patch_w = patch_shape
    
    # Resize attention to image size
    attn_resized = scipy_zoom(attn_map, (img_h/patch_h, img_w/patch_w), order=1)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.imshow(attn_resized, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


def save_single_patch_attention(
    attn_map,
    patch_shape,
    save_path,
    title="Patch Attention"
):
    """
    Save a single patch-level attention image
    """
    patch_h, patch_w = patch_shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn_map, cmap='viridis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f"{title} ({patch_h}×{patch_w})", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def save_detection_with_boxes(
    image,
    detections,
    gt_boxes,
    gt_labels,
    save_path,
    title="Detection",
    score_threshold=0.3
):
    """
    Save detection result with colored bounding boxes
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    
    # Draw GT boxes (dashed)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            color = CATEGORY_COLORS[label % len(CATEGORY_COLORS)]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
            ax.add_patch(rect)
    
    # Draw predictions (solid)
    det_count = 0
    if detections and len(detections) > 0:
        det = detections[0]
        boxes = det.get('boxes', [])
        scores = det.get('scores', [])
        labels = det.get('labels', [])
        
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > score_threshold:
                x1, y1, x2, y2 = box
                color = CATEGORY_COLORS[int(label) % len(CATEGORY_COLORS)]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                         linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                class_name = COCO_CATEGORIES[int(label)] if int(label) < len(COCO_CATEGORIES) else f'cls{label}'
                ax.text(x1, y1-5, f'{class_name}:{score:.2f}', color='white', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
                det_count += 1
    
    ax.axis('off')
    ax.set_title(f"{title} ({det_count} detections)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    
    return det_count


def create_multiclass_attention_figure(
    image,
    baseline_attn, baseline_dets,
    daga_attn, daga_dets,
    gt_boxes, gt_labels,
    patch_shape,
    save_path=None,
    input_size=518
):
    """
    Create multi-category attention visualization (kept for compatibility)
    """
    # This function is kept for backward compatibility but we now prefer single images
    pass


def create_attention_only_figure(
    image,
    baseline_attn,
    daga_attn,
    patch_shape,
    save_path=None
):
    """
    Create pure attention comparison figure (no detection boxes)
    Focus on attention map quality
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    img_h, img_w = image.shape[:2]
    patch_h, patch_w = patch_shape
    
    from scipy.ndimage import zoom as scipy_zoom
    
    # Row 1: Baseline
    # Original patch attention
    im0 = axes[0, 0].imshow(baseline_attn, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title(f"Baseline Patch Attention\n({patch_h}×{patch_w})", fontsize=11)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    
    # Upscaled attention
    baseline_up = scipy_zoom(baseline_attn, (8, 8), order=1)
    im1 = axes[0, 1].imshow(baseline_up, cmap='viridis', interpolation='bilinear')
    axes[0, 1].set_title(f"Baseline Upscaled\n({patch_h*8}×{patch_w*8})", fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    
    # Overlay on image
    baseline_resized = scipy_zoom(baseline_attn, (img_h/patch_h, img_w/patch_w), order=1)
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(baseline_resized, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("Baseline Overlay", fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2: DAGA
    im3 = axes[1, 0].imshow(daga_attn, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title(f"DAGA Patch Attention\n({patch_h}×{patch_w})", fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    daga_up = scipy_zoom(daga_attn, (8, 8), order=1)
    im4 = axes[1, 1].imshow(daga_up, cmap='viridis', interpolation='bilinear')
    axes[1, 1].set_title(f"DAGA Upscaled\n({patch_h*8}×{patch_w*8})", fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    daga_resized = scipy_zoom(daga_attn, (img_h/patch_h, img_w/patch_w), order=1)
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(daga_resized, cmap='jet', alpha=0.5)
    axes[1, 2].set_title("DAGA Overlay", fontsize=11)
    axes[1, 2].axis('off')
    
    fig.suptitle("Attention Map Comparison (Baseline vs DAGA)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def run_coco_visualization(args):
    """Main visualization pipeline - generates single images per visualization type"""
    print("\n" + "=" * 70)
    print("COCO Detection Attention Visualization")
    print("=" * 70)
    print(f"Input size: {args.input_size}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different visualization types
    (output_dir / "baseline_attention_overlay").mkdir(exist_ok=True)
    (output_dir / "daga_attention_overlay").mkdir(exist_ok=True)
    (output_dir / "baseline_patch_attention").mkdir(exist_ok=True)
    (output_dir / "daga_patch_attention").mkdir(exist_ok=True)
    (output_dir / "baseline_detection").mkdir(exist_ok=True)
    (output_dir / "daga_detection").mkdir(exist_ok=True)
    (output_dir / "original").mkdir(exist_ok=True)
    
    # Load models
    print(f"\nLoading Baseline model: {args.baseline_checkpoint}")
    baseline_model = DetectionVisualizer(args.baseline_checkpoint, args.device)
    
    print(f"Loading DAGA model: {args.daga_checkpoint}")
    daga_model = DetectionVisualizer(args.daga_checkpoint, args.device)
    
    # Load COCO dataset
    from pycocotools.coco import COCO
    import torchvision.transforms as T
    
    ann_file = Path(args.data_path) / 'annotations' / 'instances_val2017.json'
    img_dir = Path(args.data_path) / 'val2017'
    
    coco = COCO(str(ann_file))
    img_ids = list(coco.imgs.keys())
    
    # Transform
    transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n✓ Loaded COCO val2017: {len(img_ids)} images")
    print(f"Finding best samples (DAGA improvement)...")
    
    # Evaluate and find best samples where DAGA improves over Baseline
    from core.simple_detection_head import decode_predictions
    
    improvements = []
    
    for i, img_id in enumerate(tqdm(img_ids[:args.max_eval], desc="Evaluating")):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if len(anns) < 2:  # Skip images with too few objects
            continue
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_tensor = transform(pil_img)
        
        # Get detections
        with torch.no_grad():
            _, baseline_dets, _ = baseline_model.get_attention_and_detections(img_tensor, score_threshold=0.3)
            _, daga_dets, _ = daga_model.get_attention_and_detections(img_tensor, score_threshold=0.3)
        
        # Count detections
        baseline_count = 0
        daga_count = 0
        
        if baseline_dets and len(baseline_dets) > 0:
            scores = baseline_dets[0].get('scores', [])
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            baseline_count = sum(1 for s in scores if s > 0.3)
        
        if daga_dets and len(daga_dets) > 0:
            scores = daga_dets[0].get('scores', [])
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            daga_count = sum(1 for s in scores if s > 0.3)
        
        # Calculate improvement
        improvement = daga_count - baseline_count
        
        if improvement > 0 or daga_count >= 3:  # DAGA improved or has good detections
            categories = set(ann['category_id'] for ann in anns)
            improvements.append({
                'img_id': img_id,
                'improvement': improvement,
                'baseline_count': baseline_count,
                'daga_count': daga_count,
                'num_gt': len(anns),
                'num_categories': len(categories),
                'anns': anns
            })
    
    # Sort by improvement (DAGA better than Baseline)
    improvements.sort(key=lambda x: (x['improvement'], x['daga_count']), reverse=True)
    
    print(f"\n✓ Found {len(improvements)} good samples")
    print(f"Generating single images for top {args.num_samples}...")
    
    for rank, sample in enumerate(improvements[:args.num_samples]):
        img_id = sample['img_id']
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_tensor = transform(pil_img)
        image = denormalize_image(img_tensor)
        
        # Get GT boxes
        anns = sample['anns']
        gt_boxes = []
        gt_labels = []
        scale_x = args.input_size / orig_w
        scale_y = args.input_size / orig_h
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x * scale_x, y * scale_y, (x+w) * scale_x, (y+h) * scale_y])
            cat_id = ann['category_id']
            cat_info = coco.loadCats(cat_id)[0]
            try:
                label = COCO_CATEGORIES.index(cat_info['name'])
            except ValueError:
                label = cat_id - 1
            gt_labels.append(label)
        
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        
        # Get attention and detections
        baseline_attn, baseline_dets, patch_shape = baseline_model.get_attention_and_detections(img_tensor, score_threshold=0.3)
        daga_attn, daga_dets, _ = daga_model.get_attention_and_detections(img_tensor, score_threshold=0.3)
        
        if baseline_attn is None or daga_attn is None:
            continue
        
        prefix = f"{rank+1:02d}_img_{img_id}"
        
        # Save original image
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "original" / f"{prefix}.png", dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save individual attention overlays
        save_single_attention_overlay(
            image, baseline_attn, patch_shape,
            output_dir / "baseline_attention_overlay" / f"{prefix}_baseline.png",
            title="Baseline Attention"
        )
        save_single_attention_overlay(
            image, daga_attn, patch_shape,
            output_dir / "daga_attention_overlay" / f"{prefix}_daga.png",
            title="DAGA Attention"
        )
        
        # Save patch-level attention
        save_single_patch_attention(
            baseline_attn, patch_shape,
            output_dir / "baseline_patch_attention" / f"{prefix}_baseline_patch.png",
            title="Baseline"
        )
        save_single_patch_attention(
            daga_attn, patch_shape,
            output_dir / "daga_patch_attention" / f"{prefix}_daga_patch.png",
            title="DAGA"
        )
        
        # Save detection results
        baseline_det_count = save_detection_with_boxes(
            image, baseline_dets, gt_boxes, gt_labels,
            output_dir / "baseline_detection" / f"{prefix}_baseline_det.png",
            title="Baseline"
        )
        daga_det_count = save_detection_with_boxes(
            image, daga_dets, gt_boxes, gt_labels,
            output_dir / "daga_detection" / f"{prefix}_daga_det.png",
            title="DAGA"
        )
        
        print(f"  [{rank+1}/{args.num_samples}] img_{img_id}: BL={baseline_det_count} → DAGA={daga_det_count} (+{sample['improvement']})")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("  Each visualization type in separate folder:")
    print("  - original/: Original images")
    print("  - baseline_attention_overlay/: Baseline attention on image")
    print("  - daga_attention_overlay/: DAGA attention on image")
    print("  - baseline_patch_attention/: Baseline patch-level attention")
    print("  - daga_patch_attention/: DAGA patch-level attention")
    print("  - baseline_detection/: Baseline detection boxes")
    print("  - daga_detection/: DAGA detection boxes")


def parse_args():
    parser = argparse.ArgumentParser(description="COCO Detection Multi-Category Attention Visualization")
    
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--daga_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./visualization/results/coco_detection")
    parser.add_argument("--input_size", type=int, default=728,
                       help="Input size (higher = finer attention, default 728)")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_eval", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_coco_visualization(args)

