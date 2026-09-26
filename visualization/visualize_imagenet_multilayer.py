"""
ImageNet Multi-Layer Attention Visualization

This script visualizes how attention evolves across different transformer layers
for ImageNet classification, showing how DAGA helps the model focus on
discriminative object features.

Key features:
- Multi-layer attention visualization (DAGA layers: 1, 2, 10, 11)
- Side-by-side comparison of Baseline vs DAGA
- Support for higher resolution input (e.g., 518 instead of 224)
- Attention upscaling for finer-grained visualization
"""

import os
os.environ['SWANLAB_DISABLED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from PIL import Image
import sys
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3"))

from core.backbones import load_dinov3_backbone, get_attention_map


# ============================================================================
# Utility Functions
# ============================================================================

def denormalize_image(img_tensor):
    """Denormalize image tensor to [0, 1] range"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(std * img + mean, 0, 1)
    return img


def resize_attention_map(attn_map, target_size):
    """Resize attention map to target size"""
    if attn_map is None:
        return None
    h_ratio = target_size[0] / attn_map.shape[0]
    w_ratio = target_size[1] / attn_map.shape[1]
    return scipy_zoom(attn_map, (h_ratio, w_ratio), order=1)


def normalize_attention(attn_map):
    """Simple min-max normalization without aggressive enhancement"""
    attn = attn_map.copy()
    attn_min = attn.min()
    attn_max = attn.max()
    if attn_max > attn_min:
        attn = (attn - attn_min) / (attn_max - attn_min)
    return attn


def process_attention_to_map(raw_attn, num_patches, H, W):
    """Process raw attention weights to spatial attention map"""
    B, num_heads, seq_len, _ = raw_attn.shape
    
    num_registers = seq_len - num_patches - 1
    if num_registers < 0:
        num_registers = 0
    
    cls_attn_all = raw_attn[:, :, 0, 1:]
    patch_start = num_registers
    cls_attn_patches = cls_attn_all[:, :, patch_start:]
    cls_attn = cls_attn_patches.mean(dim=1)
    
    min_val = cls_attn.amin(dim=1, keepdim=True)
    max_val = cls_attn.amax(dim=1, keepdim=True)
    cls_attn_norm = (cls_attn - min_val) / (max_val - min_val + 1e-8)
    
    if cls_attn_norm.shape[1] == num_patches:
        return cls_attn_norm.reshape(B, H, W).cpu().numpy()
    return None


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract args"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = None
    if "args" in checkpoint:
        args = argparse.Namespace(**checkpoint["args"])
    return checkpoint, args


# ============================================================================
# Multi-Layer Attention Model
# ============================================================================

class MultiLayerAttentionModel(nn.Module):
    """Wrapper model that extracts attention from multiple layers"""
    
    def __init__(self, checkpoint_path, device='cuda', num_classes=1000):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        from tasks.classification import ClassificationModel
        
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vitb16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path', 
                                   'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
        
        if 'classifier.weight' in checkpoint.get('model_state_dict', {}):
            num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
        
        use_daga = getattr(self.ckpt_args, 'use_daga', False)
        daga_layers = getattr(self.ckpt_args, 'daga_layers', [])
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = ClassificationModel(
            vit_model, num_classes=num_classes,
            use_daga=use_daga, daga_layers=daga_layers
        )
        
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.num_layers = len(self.model.vit.blocks)
        
        print(f"✓ Model loaded: DAGA={use_daga}, layers={daga_layers}, total_blocks={self.num_layers}")
    
    def get_multi_layer_attention(self, x, target_layers=[1, 2, 10, 11]):
        """Extract attention maps from multiple layers"""
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        B = x.shape[0]
        vit = self.model.vit
        
        x_processed, (H, W) = vit.prepare_tokens_with_masks(x)
        num_patches = H * W
        seq_len = x_processed.shape[1]
        num_registers = seq_len - num_patches - 1
        
        layer_attentions = {}
        
        daga_guidance_map = None
        if self.use_daga:
            from core.backbones import compute_daga_guidance_map
            daga_guidance_map = compute_daga_guidance_map(
                vit, x_processed, H, W, len(vit.blocks) - 1
            )
        
        for idx, block in enumerate(vit.blocks):
            rope_sincos = vit.rope_embed(H=H, W=W) if vit.rope_embed else None
            
            if not self.use_daga and idx in target_layers:
                with torch.no_grad():
                    attn_weights = get_attention_map(block, x_processed)
                    attn_map = process_attention_to_map(attn_weights, num_patches, H, W)
                    if attn_map is not None:
                        layer_attentions[idx] = attn_map[0]
            
            x_processed = block(x_processed, rope_sincos)
            
            if self.use_daga and idx in self.daga_layers and daga_guidance_map is not None:
                cls_token = x_processed[:, :1, :]
                register_tokens = x_processed[:, 1:1+num_registers, :]
                patch_tokens = x_processed[:, 1+num_registers:, :]
                
                daga_module = self.model.daga_modules[str(idx)]
                adapted_patch_tokens = daga_module(patch_tokens, daga_guidance_map)
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                
                if idx in target_layers:
                    with torch.no_grad():
                        attn_weights = get_attention_map(block, x_processed)
                        attn_map = process_attention_to_map(attn_weights, num_patches, H, W)
                        if attn_map is not None:
                            layer_attentions[idx] = attn_map[0]
        
        x_processed = vit.norm(x_processed)
        cls_token = x_processed[:, 0]
        logits = self.model.classifier(cls_token)
        
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
        
        return layer_attentions, logits, pred_class, pred_prob, (H, W)


# ============================================================================
# ImageNet Dataset Loader
# ============================================================================

def get_imagenet_dataset(data_path, input_size=518, split='val', shuffle_seed=42):
    """Load ImageNet validation dataset with optional shuffling"""
    from torchvision import transforms, datasets
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_path = Path(data_path) / split
    if not val_path.exists():
        val_path = Path(data_path)
    
    dataset = datasets.ImageFolder(root=str(val_path), transform=transform)
    
    # Create shuffled indices for random sampling across all classes
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        shuffled_indices = np.random.permutation(len(dataset))
    else:
        shuffled_indices = np.arange(len(dataset))
    
    # Load ImageNet class names
    class_names = None
    try:
        # Try to load from imagenet_classes.json if exists
        class_file = PROJECT_ROOT / "data" / "imagenet_classes.json"
        if class_file.exists():
            with open(class_file) as f:
                class_names = json.load(f)
        else:
            # Use folder names
            class_names = dataset.classes
    except:
        class_names = dataset.classes
    
    return dataset, class_names, shuffled_indices


# ============================================================================
# Visualization Functions
# ============================================================================

def create_pure_attention_figure(
    baseline_attn,
    daga_attn,
    baseline_pred, baseline_prob,
    daga_pred, daga_prob,
    true_label,
    class_names,
    attn_shape,
    save_path=None
):
    """
    Create pure attention map visualization showing patch-level CLS attention
    Clean display without enhancement, preserving original attention distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
    baseline_name = class_names[baseline_pred] if baseline_pred < len(class_names) else f"Class {baseline_pred}"
    daga_name = class_names[daga_pred] if daga_pred < len(class_names) else f"Class {daga_pred}"
    
    baseline_correct = baseline_pred == true_label
    daga_correct = daga_pred == true_label
    
    # Simple normalization
    base_norm = normalize_attention(baseline_attn)
    daga_norm = normalize_attention(daga_attn)
    
    # Baseline attention (patch level)
    im0 = axes[0].imshow(base_norm, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f"Baseline ({attn_shape[0]}×{attn_shape[1]})\n{baseline_name[:20]}\n{baseline_prob:.3f} {'✓' if baseline_correct else '✗'}", 
                     fontsize=11, color='green' if baseline_correct else 'red')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # DAGA attention
    im1 = axes[1].imshow(daga_norm, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"DAGA ({attn_shape[0]}×{attn_shape[1]})\n{daga_name[:20]}\n{daga_prob:.3f} {'✓' if daga_correct else '✗'}", 
                     fontsize=11, color='green' if daga_correct else 'red')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Difference
    diff = daga_norm - base_norm
    max_abs = max(abs(diff.min()), abs(diff.max()), 1e-6)
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, interpolation='nearest')
    axes[2].set_title(f"Difference\nGap: {daga_prob - baseline_prob:+.3f}", fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    status = "✓ DAGA Corrected" if (daga_correct and not baseline_correct) else ""
    fig.suptitle(f"CLS-Patch Attention | GT: {true_name[:30]} | {status}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def create_overlay_attention_figure(
    image,
    baseline_attn,
    daga_attn,
    baseline_pred, baseline_prob,
    daga_pred, daga_prob,
    true_label,
    class_names,
    save_path=None
):
    """
    Create clean attention overlay visualization using patch-level display
    Attention map overlaid on original image with transparency
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
    baseline_name = class_names[baseline_pred] if baseline_pred < len(class_names) else f"Class {baseline_pred}"
    daga_name = class_names[daga_pred] if daga_pred < len(class_names) else f"Class {daga_pred}"
    
    baseline_correct = baseline_pred == true_label
    daga_correct = daga_pred == true_label
    
    img_size = image.shape[:2]
    
    # Simple normalization without aggressive enhancement
    base_attn_norm = normalize_attention(baseline_attn)
    daga_attn_norm = normalize_attention(daga_attn)
    
    # Resize to image size using nearest neighbor (preserve patch structure)
    base_attn_resized = resize_attention_map(base_attn_norm, img_size)
    daga_attn_resized = resize_attention_map(daga_attn_norm, img_size)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original\nGT: {true_name[:25]}", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Baseline overlay with fixed alpha
    axes[1].imshow(image)
    axes[1].imshow(base_attn_resized, cmap='jet', alpha=0.5)
    color = 'green' if baseline_correct else 'red'
    axes[1].set_title(f"Baseline\n{baseline_name[:20]}\n{baseline_prob:.3f} {'✓' if baseline_correct else '✗'}", 
                     fontsize=11, color=color)
    axes[1].axis('off')
    
    # DAGA overlay
    axes[2].imshow(image)
    axes[2].imshow(daga_attn_resized, cmap='jet', alpha=0.5)
    color = 'green' if daga_correct else 'red'
    axes[2].set_title(f"DAGA\n{daga_name[:20]}\n{daga_prob:.3f} {'✓' if daga_correct else '✗'}", 
                     fontsize=11, color=color)
    axes[2].axis('off')
    
    # Difference
    diff = daga_attn_resized - base_attn_resized
    max_abs = max(abs(diff.min()), abs(diff.max()), 1e-6)
    axes[3].imshow(image)
    axes[3].imshow(diff, cmap='RdBu_r', alpha=0.6, vmin=-max_abs, vmax=max_abs)
    axes[3].set_title(f"Difference\nGap: {daga_prob - baseline_prob:+.3f}", fontsize=11)
    axes[3].axis('off')
    
    status = "✓ DAGA Corrected" if (daga_correct and not baseline_correct) else \
             ("Both Correct" if daga_correct and baseline_correct else "")
    fig.suptitle(f"Attention (Layer 12) | {status}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def create_multilayer_comparison_figure(
    image, 
    baseline_attns, 
    daga_attns,
    baseline_pred, baseline_prob,
    daga_pred, daga_prob,
    true_label,
    class_names,
    layer_indices,
    save_path=None
):
    """Create comprehensive multi-layer attention comparison figure"""
    num_layers = len(layer_indices)
    
    fig = plt.figure(figsize=(16, 4 + num_layers * 3))
    gs = GridSpec(num_layers + 1, 4, figure=fig, height_ratios=[1.2] + [1] * num_layers,
                  hspace=0.3, wspace=0.15)
    
    # Row 0: Original image and info
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(image)
    true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
    ax_img.set_title(f"Original Image\nGround Truth: {true_name}", fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    ax_info = fig.add_subplot(gs[0, 2:])
    ax_info.axis('off')
    
    baseline_correct = baseline_pred == true_label
    daga_correct = daga_pred == true_label
    
    baseline_name = class_names[baseline_pred] if baseline_pred < len(class_names) else f"Class {baseline_pred}"
    daga_name = class_names[daga_pred] if daga_pred < len(class_names) else f"Class {daga_pred}"
    
    info_text = f"""
    Baseline Prediction: {baseline_name}
    Confidence: {baseline_prob:.3f}
    {'✓ Correct' if baseline_correct else '✗ Wrong'}
    
    DAGA Prediction: {daga_name}
    Confidence: {daga_prob:.3f}  
    {'✓ Correct' if daga_correct else '✗ Wrong'}
    
    Confidence Gain: {daga_prob - baseline_prob:+.3f}
    """
    
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    layer_names = {
        1: "Layer 2\n(DAGA Early)",
        2: "Layer 3\n(DAGA Early)", 
        10: "Layer 11\n(DAGA Late)",
        11: "Layer 12\n(DAGA Final)",
    }
    
    img_size = image.shape[:2]
    
    for row_idx, layer_idx in enumerate(layer_indices):
        ax_base = fig.add_subplot(gs[row_idx + 1, :2])
        
        if layer_idx in baseline_attns:
            base_attn = resize_attention_map(baseline_attns[layer_idx], img_size)
            ax_base.imshow(image)
            ax_base.imshow(base_attn, cmap='jet', alpha=0.6)
        else:
            ax_base.imshow(image)
            ax_base.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_base.transAxes)
        
        layer_name = layer_names.get(layer_idx, f"Layer {layer_idx + 1}")
        ax_base.set_title(f"Baseline - {layer_name}", fontsize=10)
        ax_base.axis('off')
        
        ax_daga = fig.add_subplot(gs[row_idx + 1, 2:])
        
        if layer_idx in daga_attns:
            daga_attn = resize_attention_map(daga_attns[layer_idx], img_size)
            ax_daga.imshow(image)
            ax_daga.imshow(daga_attn, cmap='jet', alpha=0.6)
        else:
            ax_daga.imshow(image)
            ax_daga.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_daga.transAxes)
        
        ax_daga.set_title(f"DAGA - {layer_name}", fontsize=10)
        ax_daga.axis('off')
    
    improvement = "✓ DAGA Improved" if (daga_correct and not baseline_correct) else \
                  ("Both Correct" if daga_correct else "Both Wrong")
    fig.suptitle(f"ImageNet Multi-Layer Attention: {true_name[:30]} | {improvement}", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


# ============================================================================
# Main Visualization Pipeline
# ============================================================================

def run_imagenet_visualization(args):
    """Main visualization pipeline for ImageNet dataset"""
    
    print("\n" + "=" * 70)
    print("ImageNet Multi-Layer Attention Visualization")
    print("=" * 70)
    print(f"Input size: {args.input_size} (trained on 224, upscaled for finer attention)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_layers = args.target_layers if args.target_layers else [1, 2, 10, 11]
    
    print(f"\nLoading Baseline model from: {args.baseline_checkpoint}")
    baseline_model = MultiLayerAttentionModel(args.baseline_checkpoint, num_classes=1000)
    
    print(f"\nLoading DAGA model from: {args.daga_checkpoint}")
    daga_model = MultiLayerAttentionModel(args.daga_checkpoint, num_classes=1000)
    
    print(f"\nTarget layers for visualization: {target_layers}")
    
    print(f"\nLoading ImageNet dataset from: {args.data_path}")
    dataset, class_names, shuffled_indices = get_imagenet_dataset(args.data_path, args.input_size)
    print(f"✓ Loaded {len(dataset)} validation samples, {len(class_names)} classes")
    print(f"  (Using random sampling across all classes for fair evaluation)")
    
    print(f"\nEvaluating samples to find best improvements...")
    
    improvements = []
    max_eval = min(args.max_eval, len(dataset))
    
    for i in tqdm(range(max_eval), desc="Evaluating"):
        idx = shuffled_indices[i]  # Use shuffled index
        img_tensor, label = dataset[idx]
        
        with torch.no_grad():
            _, _, baseline_pred, baseline_prob, _ = baseline_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
            _, _, daga_pred, daga_prob, _ = daga_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
        
        baseline_correct = baseline_pred == label
        daga_correct = daga_pred == label
        
        if daga_correct and not baseline_correct:
            conf_gap = daga_prob - baseline_prob
            improvement_score = 10.0 + daga_prob + max(0, conf_gap)
        elif daga_correct and baseline_correct:
            improvement_score = daga_prob - baseline_prob
        else:
            improvement_score = -10.0
        
        improvements.append({
            'idx': idx,
            'label': label,
            'baseline_pred': baseline_pred,
            'baseline_prob': baseline_prob,
            'baseline_correct': baseline_correct,
            'daga_pred': daga_pred,
            'daga_prob': daga_prob,
            'daga_correct': daga_correct,
            'improvement': improvement_score,
            'conf_gap': daga_prob - baseline_prob
        })
    
    # Filter: Only DAGA correct AND Baseline wrong
    corrected_samples = [x for x in improvements if x['daga_correct'] and not x['baseline_correct']]
    
    # Sort by confidence gap (largest gap first)
    corrected_samples.sort(key=lambda x: x['conf_gap'], reverse=True)
    
    print(f"\n✓ Found {len(corrected_samples)} samples where DAGA corrected Baseline's mistake")
    
    if corrected_samples:
        print("\nTop corrected samples sorted by confidence gap (DAGA ✓, Baseline ✗):")
        for i, item in enumerate(corrected_samples[:10]):
            true_name = class_names[item['label']] if item['label'] < len(class_names) else f"Class {item['label']}"
            print(f"  {i+1}. {true_name[:30]}: gap={item['conf_gap']:+.3f} "
                  f"(BL: {item['baseline_prob']:.3f} -> DAGA: {item['daga_prob']:.3f})")
    
    print(f"\nGenerating visualizations for top {args.num_samples} corrected samples...")
    
    # Create output directories
    (output_dir / "pure_attention").mkdir(exist_ok=True)
    (output_dir / "overlay_attention").mkdir(exist_ok=True)
    (output_dir / "multilayer").mkdir(exist_ok=True)
    
    # Only visualize last layer (layer 11) for pure/overlay, all layers for multilayer
    last_layer = 11
    
    for rank, item in enumerate(corrected_samples[:args.num_samples]):
        idx = item['idx']
        img_tensor, label = dataset[idx]
        image = denormalize_image(img_tensor)
        
        with torch.no_grad():
            # Get attention for last layer only (for pure/overlay)
            baseline_attns, _, baseline_pred, baseline_prob, (H, W) = baseline_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), [last_layer]
            )
            daga_attns, _, daga_pred, daga_prob, _ = daga_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), [last_layer]
            )
        
        true_class = class_names[label] if label < len(class_names) else f"class{label}"
        baseline_pred_name = class_names[item['baseline_pred']] if item['baseline_pred'] < len(class_names) else f"class{item['baseline_pred']}"
        daga_pred_name = class_names[item['daga_pred']] if item['daga_pred'] < len(class_names) else f"class{item['daga_pred']}"
        
        # Sanitize class names for filename
        true_class_safe = true_class.replace('/', '_').replace(' ', '_')[:20]
        baseline_safe = baseline_pred_name.replace('/', '_').replace(' ', '_')[:15]
        daga_safe = daga_pred_name.replace('/', '_').replace(' ', '_')[:15]
        
        # Get last layer attention
        base_attn_last = baseline_attns.get(last_layer)
        daga_attn_last = daga_attns.get(last_layer)
        
        if base_attn_last is not None and daga_attn_last is not None:
            # 1. Pure attention map (no image overlay)
            pure_path = output_dir / "pure_attention" / f"{rank+1:02d}_{true_class_safe}_gap{item['conf_gap']:+.3f}_pure.png"
            create_pure_attention_figure(
                base_attn_last, daga_attn_last,
                baseline_pred, baseline_prob,
                daga_pred, daga_prob,
                label, class_names, (H, W),
                save_path=pure_path
            )
            
            # 2. Overlay attention (heatmap on original image)
            overlay_path = output_dir / "overlay_attention" / f"{rank+1:02d}_{true_class_safe}_gap{item['conf_gap']:+.3f}_overlay.png"
            create_overlay_attention_figure(
                image, base_attn_last, daga_attn_last,
                baseline_pred, baseline_prob,
                daga_pred, daga_prob,
                label, class_names,
                save_path=overlay_path
            )
        
        # 3. Multi-layer comparison (optional, for full analysis)
        if args.save_multilayer:
            with torch.no_grad():
                baseline_attns_full, _, _, _, _ = baseline_model.get_multi_layer_attention(
                    img_tensor.unsqueeze(0), target_layers
                )
                daga_attns_full, _, _, _, _ = daga_model.get_multi_layer_attention(
                    img_tensor.unsqueeze(0), target_layers
                )
            
            multi_path = output_dir / "multilayer" / f"{rank+1:02d}_{true_class_safe}_gap{item['conf_gap']:+.3f}_multilayer.png"
            create_multilayer_comparison_figure(
                image, baseline_attns_full, daga_attns_full,
                baseline_pred, baseline_prob,
                daga_pred, daga_prob,
                label, class_names, target_layers,
                save_path=multi_path
            )
        
        print(f"  [{rank+1}/{args.num_samples}] {true_class[:25]}: "
              f"BL={baseline_pred_name[:15]} ({baseline_prob:.3f}) -> "
              f"DAGA={daga_pred_name[:15]} ({daga_prob:.3f}), gap={item['conf_gap']:+.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    baseline_acc = sum(1 for x in improvements if x['baseline_correct']) / len(improvements)
    daga_acc = sum(1 for x in improvements if x['daga_correct']) / len(improvements)
    corrected = len(corrected_samples)
    
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"DAGA Accuracy: {daga_acc:.2%}")
    print(f"Samples corrected by DAGA: {corrected}")
    print(f"\nResults saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet Multi-Layer Attention Visualization")
    
    parser.add_argument("--baseline_checkpoint", type=str, required=True,
                       help="Path to baseline model checkpoint")
    parser.add_argument("--daga_checkpoint", type=str, required=True,
                       help="Path to DAGA model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to ImageNet dataset")
    parser.add_argument("--output_dir", type=str, default="./visualization/results/imagenet_multilayer",
                       help="Output directory for visualizations")
    parser.add_argument("--input_size", type=int, default=518,
                       help="Input image size (use higher for finer attention maps: 518, 728, 1024)")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to visualize")
    parser.add_argument("--max_eval", type=int, default=5000,
                       help="Maximum samples to evaluate (use larger for accurate stats)")
    parser.add_argument("--target_layers", type=int, nargs="+", default=None,
                       help="Layers to visualize (default: 1 2 10 11)")
    parser.add_argument("--save_multilayer", action="store_true",
                       help="Also save multi-layer comparison figures")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_imagenet_visualization(args)

