"""
DTD (Describable Textures Dataset) Multi-Layer Attention Visualization

This script visualizes how attention evolves across different transformer layers
for texture classification, showing how DAGA helps the model focus on 
discriminative texture units at different levels of abstraction.

Key features:
- Multi-layer attention visualization (layers 1, 4, 8, 11)
- Side-by-side comparison of Baseline vs DAGA
- Shows attention progression from low-level to high-level features
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


def process_attention_to_map(raw_attn, num_patches, H, W):
    """Process raw attention weights to spatial attention map"""
    # raw_attn: [B, heads, seq_len, seq_len]
    B, num_heads, seq_len, _ = raw_attn.shape
    
    # Calculate number of register tokens
    num_registers = seq_len - num_patches - 1
    if num_registers < 0:
        num_registers = 0
    
    # Get CLS attention to all other tokens (excluding CLS itself)
    cls_attn_all = raw_attn[:, :, 0, 1:]  # [B, heads, seq_len-1]
    
    # Skip register tokens, get only patch attention
    patch_start = num_registers
    cls_attn_patches = cls_attn_all[:, :, patch_start:]  # [B, heads, num_patches]
    
    # Average over heads
    cls_attn = cls_attn_patches.mean(dim=1)  # [B, num_patches]
    
    # Normalize
    min_val = cls_attn.amin(dim=1, keepdim=True)
    max_val = cls_attn.amax(dim=1, keepdim=True)
    cls_attn_norm = (cls_attn - min_val) / (max_val - min_val + 1e-8)
    
    # Reshape to spatial
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
# Multi-Layer Attention Extractor
# ============================================================================

class MultiLayerAttentionModel(nn.Module):
    """
    Wrapper model that extracts attention from multiple layers
    """
    def __init__(self, checkpoint_path, device='cuda', num_classes=47):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Load model
        from tasks.classification import ClassificationModel
        
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vitl16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path', 
                                   'checkpoints/dinov3_vitl16_pretrain_lvd1689m.pth')
        
        # Get num_classes from checkpoint
        if 'classifier.weight' in checkpoint.get('model_state_dict', {}):
            num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
        
        use_daga = getattr(self.ckpt_args, 'use_daga', False)
        daga_layers = getattr(self.ckpt_args, 'daga_layers', [])
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = ClassificationModel(
            vit_model, num_classes=num_classes,
            use_daga=use_daga, daga_layers=daga_layers
        )
        
        # Load weights
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
    
    def get_multi_layer_attention(self, x, target_layers=[0, 3, 7, 11]):
        """
        Extract attention maps from multiple layers
        
        For Baseline: captures attention at each target layer (before block forward)
        For DAGA: captures attention AFTER DAGA modification using the SAME block
                  to show how DAGA-modified features change the attention pattern
        
        Args:
            x: Input image tensor [B, 3, H, W]
            target_layers: List of layer indices to extract attention from
        
        Returns:
            dict: {layer_idx: attention_map} for each target layer
            logits: Classification logits
            pred_class: Predicted class index
            pred_prob: Prediction probability
        """
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        B = x.shape[0]
        vit = self.model.vit
        
        # Prepare tokens
        x_processed, (H, W) = vit.prepare_tokens_with_masks(x)
        num_patches = H * W
        seq_len = x_processed.shape[1]
        num_registers = seq_len - num_patches - 1
        
        # Storage for attention maps
        layer_attentions = {}
        
        # Compute guidance map for DAGA (from last layer)
        daga_guidance_map = None
        if self.use_daga:
            from core.backbones import compute_daga_guidance_map
            daga_guidance_map = compute_daga_guidance_map(
                vit, x_processed, H, W, len(vit.blocks) - 1
            )
        
        # Forward through blocks - matching ClassificationModel.forward() exactly
        for idx, block in enumerate(vit.blocks):
            rope_sincos = vit.rope_embed(H=H, W=W) if vit.rope_embed else None
            
            # Step 1: For BASELINE model, capture attention BEFORE block forward
            # This is what the original code does for baseline_attn_weights
            if not self.use_daga and idx in target_layers:
                with torch.no_grad():
                    attn_weights = get_attention_map(block, x_processed)
                    attn_map = process_attention_to_map(attn_weights, num_patches, H, W)
                    if attn_map is not None:
                        layer_attentions[idx] = attn_map[0]
            
            # Step 2: Block forward
            x_processed = block(x_processed, rope_sincos)
            
            # Step 3: Apply DAGA AFTER block forward
            if self.use_daga and idx in self.daga_layers and daga_guidance_map is not None:
                cls_token = x_processed[:, :1, :]
                register_tokens = x_processed[:, 1:1+num_registers, :]
                patch_tokens = x_processed[:, 1+num_registers:, :]
                
                # Apply DAGA
                daga_module = self.model.daga_modules[str(idx)]
                adapted_patch_tokens = daga_module(patch_tokens, daga_guidance_map)
                
                # Reconstruct
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                
                # Step 4: For DAGA model, capture attention AFTER DAGA using SAME block
                # This shows how DAGA-modified features change the attention
                if idx in target_layers:
                    with torch.no_grad():
                        attn_weights = get_attention_map(block, x_processed)
                        attn_map = process_attention_to_map(attn_weights, num_patches, H, W)
                        if attn_map is not None:
                            layer_attentions[idx] = attn_map[0]
        
        # Final classification
        x_processed = vit.norm(x_processed)
        cls_token = x_processed[:, 0]
        logits = self.model.classifier(cls_token)
        
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
        
        return layer_attentions, logits, pred_class, pred_prob


# ============================================================================
# DTD Dataset Loader
# ============================================================================

class DTDDataset(torch.utils.data.Dataset):
    """Describable Textures Dataset (47 classes) - Custom loader"""
    def __init__(self, root, split="test", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Use split1 by default
        labels_dir = self.root / "labels"
        images_dir = self.root / "images"
        
        # Load split file
        split_file = labels_dir / f"{split}1.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"DTD split file not found: {split_file}")
        
        # Build class list from image paths
        class_names = set()
        with open(split_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                class_name = rel_path.split("/")[0]
                class_names.add(class_name)
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load samples
        with open(split_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                class_name = rel_path.split("/")[0]
                img_path = images_dir / rel_path
                
                if img_path.exists():
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)
        
        print(f"DTD {split}: loaded {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


def get_dtd_dataset(data_path, input_size=518):
    """Load DTD dataset for visualization"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DTDDataset(root=data_path, split='test', transform=transform)
    class_names = dataset.classes
    
    return dataset, class_names


# ============================================================================
# Visualization Functions
# ============================================================================

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
    """
    Create a comprehensive multi-layer attention comparison figure
    
    Layout:
    Row 0: Original Image | Ground Truth Info
    Row 1-4: Layer attentions (Baseline | DAGA) for each target layer
    """
    num_layers = len(layer_indices)
    
    # Figure setup
    fig = plt.figure(figsize=(16, 4 + num_layers * 3))
    
    # Create grid: 1 row for image + (num_layers) rows for attention
    gs = GridSpec(num_layers + 1, 4, figure=fig, height_ratios=[1.2] + [1] * num_layers,
                  hspace=0.3, wspace=0.15)
    
    # Row 0: Original image and info
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(image)
    ax_img.set_title(f"Original Image\nGround Truth: {class_names[true_label]}", fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Prediction info
    ax_info = fig.add_subplot(gs[0, 2:])
    ax_info.axis('off')
    
    baseline_correct = baseline_pred == true_label
    daga_correct = daga_pred == true_label
    
    info_text = f"""
    Baseline Prediction: {class_names[baseline_pred]}
    Confidence: {baseline_prob:.3f}
    {'✓ Correct' if baseline_correct else '✗ Wrong'}
    
    DAGA Prediction: {class_names[daga_pred]}
    Confidence: {daga_prob:.3f}  
    {'✓ Correct' if daga_correct else '✗ Wrong'}
    
    Confidence Gain: {daga_prob - baseline_prob:+.3f}
    """
    
    # Color based on improvement
    text_color = 'green' if daga_correct and not baseline_correct else ('blue' if daga_correct else 'red')
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Layer attention rows
    layer_names = {
        # DAGA layers for ViT-S/B (12 layers)
        1: "Layer 2\n(DAGA Early)",
        2: "Layer 3\n(DAGA Early)", 
        10: "Layer 11\n(DAGA Late)",
        11: "Layer 12\n(DAGA Final)",
        # For larger models (24 layers)
        22: "Layer 23\n(DAGA Late)",
        23: "Layer 24\n(DAGA Final)",
    }
    
    img_size = image.shape[:2]
    
    for row_idx, layer_idx in enumerate(layer_indices):
        # Baseline attention
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
        
        # DAGA attention
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
    
    # Main title
    improvement = "✓ DAGA Improved" if (daga_correct and not baseline_correct) else \
                  ("Both Correct" if daga_correct else "Both Wrong")
    fig.suptitle(f"DTD Multi-Layer Attention: {class_names[true_label]} | {improvement}", 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Use subplots_adjust instead of tight_layout to avoid warning with GridSpec
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
    
    return fig


def create_attention_evolution_figure(
    image,
    attentions,  # dict of {layer_idx: attn_map}
    model_name,
    pred_class,
    pred_prob,
    true_label,
    class_names,
    layer_indices,
    save_path=None
):
    """
    Create a single-model attention evolution figure showing how attention
    changes across layers for texture recognition
    """
    num_layers = len(layer_indices)
    
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
    
    img_size = image.shape[:2]
    
    # Original image
    axes[0].imshow(image)
    correct = pred_class == true_label
    color = 'green' if correct else 'red'
    axes[0].set_title(f"Original\nTrue: {class_names[true_label]}\nPred: {class_names[pred_class]}\nConf: {pred_prob:.3f}",
                     fontsize=9, color=color)
    axes[0].axis('off')
    
    # Layer attentions
    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i + 1]
        
        if layer_idx in attentions:
            attn = resize_attention_map(attentions[layer_idx], img_size)
            ax.imshow(image)
            ax.imshow(attn, cmap='jet', alpha=0.6)
        else:
            ax.imshow(image)
        
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)
        ax.axis('off')
    
    fig.suptitle(f"{model_name} Attention Evolution", fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98, wspace=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


# ============================================================================
# Quantitative Analysis Visualization
# ============================================================================

def create_quantitative_visualizations(attention_stats, target_layers, output_dir):
    """
    Create visualizations for quantitative attention analysis
    
    Generates:
    1. Bar chart: Average metrics by layer
    2. Heatmap: Sample × Layer matrix for each metric
    3. Box plot: Distribution of metrics by layer
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(attention_stats)
    
    # Create output directory for quantitative plots
    quant_dir = output_dir / "quantitative_analysis"
    quant_dir.mkdir(exist_ok=True)
    
    # Define metrics and their display names
    metrics = {
        'entropy_change': ('Entropy Change', 'RdYlGn_r'),  # Red=increase, Green=decrease
        'gini_change': ('Gini Coefficient Change', 'RdYlGn'),  # Green=increase (more focused)
        'l2_distance': ('L2 Distance', 'YlOrRd'),  # Magnitude of change
        'cosine_similarity': ('Cosine Similarity', 'RdYlGn'),  # Green=similar
    }
    
    # ========================================================================
    # 1. Bar Chart: Average Metrics by Layer
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    layer_labels = [f"Layer {l+1}" for l in target_layers]
    
    for idx, (metric_key, (metric_name, _)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Calculate mean and std by layer
        layer_means = []
        layer_stds = []
        for layer in target_layers:
            layer_data = df[df['layer'] == layer][metric_key]
            layer_means.append(layer_data.mean())
            layer_stds.append(layer_data.std())
        
        # Create bar chart
        bars = ax.bar(layer_labels, layer_means, yerr=layer_stds, capsize=5,
                     color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(target_layers)],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels on bars
        for bar, mean in zip(bars, layer_means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Layer', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('DAGA vs Baseline: Attention Metrics by Layer\n(Corrected Samples Only)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(quant_dir / 'metrics_by_layer_barplot.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_by_layer_barplot.png")
    
    # ========================================================================
    # 2. Heatmap: Sample × Layer Matrix
    # ========================================================================
    for metric_key, (metric_name, cmap) in metrics.items():
        # Pivot table: samples as rows, layers as columns
        unique_samples = df['sample_idx'].unique()
        
        if len(unique_samples) > 1:
            # Create matrix
            matrix = np.zeros((len(unique_samples), len(target_layers)))
            sample_labels = []
            
            for i, sample_idx in enumerate(unique_samples):
                sample_data = df[df['sample_idx'] == sample_idx]
                sample_labels.append(sample_data.iloc[0]['true_class'][:10])  # Truncate class name
                for j, layer in enumerate(target_layers):
                    layer_data = sample_data[sample_data['layer'] == layer]
                    if len(layer_data) > 0:
                        matrix[i, j] = layer_data[metric_key].values[0]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, max(8, len(unique_samples) * 0.4)))
            
            im = ax.imshow(matrix, aspect='auto', cmap=cmap)
            
            # Labels
            ax.set_xticks(np.arange(len(target_layers)))
            ax.set_xticklabels([f'L{l+1}' for l in target_layers], fontsize=11)
            ax.set_yticks(np.arange(len(unique_samples)))
            ax.set_yticklabels(sample_labels, fontsize=9)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(metric_name, fontsize=11)
            
            # Add value annotations
            for i in range(len(unique_samples)):
                for j in range(len(target_layers)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                  ha='center', va='center', fontsize=8,
                                  color='white' if abs(matrix[i, j]) > abs(matrix).max()/2 else 'black')
            
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Sample (True Class)', fontsize=12)
            ax.set_title(f'{metric_name}\n(Sample × Layer Heatmap)', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(quant_dir / f'heatmap_{metric_key}.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✓ Saved: heatmap_{metric_key}.png")
    
    # ========================================================================
    # 3. Box Plot: Distribution by Layer
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (metric_key, (metric_name, _)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Prepare data for box plot
        box_data = [df[df['layer'] == layer][metric_key].values for layer in target_layers]
        
        bp = ax.boxplot(box_data, labels=[f'L{l+1}' for l in target_layers],
                       patch_artist=True, notch=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(target_layers)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('DAGA vs Baseline: Metric Distributions by Layer\n(Corrected Samples Only)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(quant_dir / 'metrics_distribution_boxplot.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_distribution_boxplot.png")
    
    # ========================================================================
    # 4. Summary Radar Chart
    # ========================================================================
    # Normalize metrics for radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    metric_names = ['Entropy Δ', 'Gini Δ', 'L2 Dist', 'Cos Sim']
    num_metrics = len(metric_names)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Plot for each layer
    for i, layer in enumerate(target_layers):
        layer_data = df[df['layer'] == layer]
        
        # Get normalized values (scale to 0-1 range for visualization)
        values = [
            layer_data['entropy_change'].mean(),
            layer_data['gini_change'].mean(),
            layer_data['l2_distance'].mean() / 10,  # Scale down
            layer_data['cosine_similarity'].mean()
        ]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Layer {layer+1}',
               color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_title('Attention Metrics Radar Chart by Layer', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(quant_dir / 'metrics_radar_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_radar_chart.png")
    
    print(f"\n✓ All quantitative visualizations saved to: {quant_dir}")


# ============================================================================
# Main Visualization Pipeline
# ============================================================================

def run_dtd_visualization(args):
    """Main visualization pipeline for DTD dataset"""
    
    print("\n" + "=" * 70)
    print("DTD Multi-Layer Attention Visualization")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine target layers based on model size
    # ViT-S/B: 12 layers, ViT-L: 24 layers
    if args.target_layers:
        target_layers = args.target_layers
    else:
        # Default layers for different model sizes
        # Will be adjusted after loading model
        target_layers = [0, 3, 7, 11]  # For 12-layer model
    
    # Load models
    print(f"\nLoading Baseline model from: {args.baseline_checkpoint}")
    baseline_model = MultiLayerAttentionModel(args.baseline_checkpoint, num_classes=47)
    
    print(f"\nLoading DAGA model from: {args.daga_checkpoint}")
    daga_model = MultiLayerAttentionModel(args.daga_checkpoint, num_classes=47)
    
    # Adjust target layers based on actual model size
    # Use DAGA layers directly to show DAGA's effect at each modified layer
    num_blocks = baseline_model.num_layers
    if num_blocks == 24:  # ViT-L
        target_layers = [1, 2, 10, 11, 22, 23]  # Early + Middle + Late DAGA layers
    elif num_blocks == 12:  # ViT-S/B
        target_layers = [1, 2, 10, 11]  # Match DAGA layers exactly
    
    print(f"\nTarget layers for visualization: {target_layers}")
    print(f"Model has {num_blocks} transformer blocks")
    
    # Load dataset
    print(f"\nLoading DTD dataset from: {args.data_path}")
    dataset, class_names = get_dtd_dataset(args.data_path, args.input_size)
    print(f"✓ Loaded {len(dataset)} test samples, {len(class_names)} classes")
    
    # Find samples where DAGA improves over baseline
    print(f"\nEvaluating samples to find best improvements...")
    
    improvements = []
    
    max_eval = min(args.max_eval, len(dataset))
    
    for idx in tqdm(range(max_eval), desc="Evaluating"):
        img_tensor, label = dataset[idx]
        
        with torch.no_grad():
            # Get predictions from both models
            _, baseline_logits, baseline_pred, baseline_prob = baseline_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
            _, daga_logits, daga_pred, daga_prob = daga_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
        
        baseline_correct = baseline_pred == label
        daga_correct = daga_pred == label
        
        # Calculate improvement score
        # Priority: DAGA correct + Baseline wrong + large confidence gap
        if daga_correct and not baseline_correct:
            # Best case: DAGA corrected the prediction
            # Score = base bonus + DAGA confidence + confidence gap
            conf_gap = daga_prob - baseline_prob
            improvement_score = 10.0 + daga_prob + max(0, conf_gap)
        elif daga_correct and baseline_correct:
            # Both correct, rank by confidence improvement
            improvement_score = daga_prob - baseline_prob
        else:
            # DAGA wrong
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
    
    # Sort by improvement
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    # Count corrected samples (DAGA correct, Baseline wrong)
    corrected_samples = [x for x in improvements if x['daga_correct'] and not x['baseline_correct']]
    print(f"\n✓ Found {len(corrected_samples)} samples where DAGA corrected Baseline's mistake")
    
    # Show top corrected samples with confidence gap
    if corrected_samples:
        print("\nTop corrected samples (DAGA ✓, Baseline ✗):")
        for i, item in enumerate(corrected_samples[:10]):
            print(f"  {i+1}. {class_names[item['label']]}: "
                  f"Baseline={class_names[item['baseline_pred']]} ({item['baseline_prob']:.3f}) -> "
                  f"DAGA=✓ ({item['daga_prob']:.3f}), gap={item['conf_gap']:+.3f}")
    
    # Visualize top improvements
    print(f"\nGenerating visualizations for top {args.num_samples} improvements...")
    
    (output_dir / "best_improvement").mkdir(exist_ok=True)
    (output_dir / "attention_evolution").mkdir(exist_ok=True)
    
    for rank, item in enumerate(improvements[:args.num_samples]):
        idx = item['idx']
        img_tensor, label = dataset[idx]
        image = denormalize_image(img_tensor)
        
        # Get multi-layer attention
        with torch.no_grad():
            baseline_attns, _, baseline_pred, baseline_prob = baseline_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
            daga_attns, _, daga_pred, daga_prob = daga_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
        
        # Create comparison figure with detailed filename
        # Format: rank_TrueClass_BaselinePred_DAGAPred_gap
        true_class = class_names[label]
        baseline_pred_name = class_names[item['baseline_pred']]
        daga_pred_name = class_names[item['daga_pred']]
        corrected_tag = "CORRECTED_" if (item['daga_correct'] and not item['baseline_correct']) else ""
        
        # Filename: rank_CORRECTED_TrueClass_BL-pred_DAGA-pred_gap
        save_path = output_dir / "best_improvement" / f"{rank+1:02d}_{corrected_tag}{true_class}_BL-{baseline_pred_name}_DAGA-{daga_pred_name}_gap{item['conf_gap']:+.3f}.png"
        create_multilayer_comparison_figure(
            image, baseline_attns, daga_attns,
            baseline_pred, baseline_prob,
            daga_pred, daga_prob,
            label, class_names, target_layers,
            save_path=save_path
        )
        
        # Create individual evolution figures with prediction info in filename
        base_evo_path = output_dir / "attention_evolution" / f"{rank+1:02d}_baseline_{true_class}_pred-{baseline_pred_name}.png"
        create_attention_evolution_figure(
            image, baseline_attns, f"Baseline (Pred: {baseline_pred_name})",
            baseline_pred, baseline_prob, label, class_names, target_layers,
            save_path=base_evo_path
        )
        
        daga_evo_path = output_dir / "attention_evolution" / f"{rank+1:02d}_daga_{true_class}_pred-{daga_pred_name}.png"
        create_attention_evolution_figure(
            image, daga_attns, f"DAGA (Pred: {daga_pred_name})",
            daga_pred, daga_prob, label, class_names, target_layers,
            save_path=daga_evo_path
        )
        
        corrected_mark = "✓CORRECTED" if (item['daga_correct'] and not item['baseline_correct']) else ""
        print(f"  [{rank+1}/{args.num_samples}] {class_names[label]} {corrected_mark}: "
              f"Baseline={class_names[baseline_pred]} ({baseline_prob:.3f}) -> "
              f"DAGA={class_names[daga_pred]} ({daga_prob:.3f}), gap={item['conf_gap']:+.3f}")
    
    # ========================================================================
    # Quantitative Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Quantitative Analysis - Attention Statistics")
    print("=" * 70)
    
    # Collect attention statistics for all evaluated samples
    print("\nComputing attention statistics for corrected samples...")
    
    attention_stats = []
    corrected_samples = [x for x in improvements if x['daga_correct'] and not x['baseline_correct']]
    
    for item in tqdm(corrected_samples[:min(50, len(corrected_samples))], desc="Analyzing"):
        idx = item['idx']
        img_tensor, label = dataset[idx]
        
        with torch.no_grad():
            baseline_attns, _, _, _ = baseline_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
            daga_attns, _, _, _ = daga_model.get_multi_layer_attention(
                img_tensor.unsqueeze(0), target_layers
            )
        
        for layer_idx in target_layers:
            if layer_idx in baseline_attns and layer_idx in daga_attns:
                base_attn = baseline_attns[layer_idx]
                daga_attn = daga_attns[layer_idx]
                
                # Normalize to probability distribution
                base_flat = base_attn.flatten()
                daga_flat = daga_attn.flatten()
                base_prob = base_flat / (base_flat.sum() + 1e-8)
                daga_prob = daga_flat / (daga_flat.sum() + 1e-8)
                
                # 1. Entropy (lower = more focused)
                base_entropy = -np.sum(base_prob * np.log(base_prob + 1e-8))
                daga_entropy = -np.sum(daga_prob * np.log(daga_prob + 1e-8))
                
                # 2. Peak value (higher = more focused)
                base_peak = base_attn.max()
                daga_peak = daga_attn.max()
                
                # 3. Gini coefficient (higher = more unequal/focused)
                def gini(arr):
                    sorted_arr = np.sort(arr.flatten())
                    n = len(sorted_arr)
                    cumsum = np.cumsum(sorted_arr)
                    return (2 * np.sum((np.arange(1, n+1) * sorted_arr)) / (n * np.sum(sorted_arr))) - (n + 1) / n
                
                base_gini = gini(base_attn)
                daga_gini = gini(daga_attn)
                
                # 4. L2 distance between attention maps
                l2_dist = np.sqrt(np.sum((base_attn - daga_attn) ** 2))
                
                # 5. Cosine similarity
                cos_sim = np.dot(base_flat, daga_flat) / (np.linalg.norm(base_flat) * np.linalg.norm(daga_flat) + 1e-8)
                
                attention_stats.append({
                    'sample_idx': idx,
                    'layer': layer_idx,
                    'true_class': class_names[label],
                    'baseline_entropy': base_entropy,
                    'daga_entropy': daga_entropy,
                    'entropy_change': daga_entropy - base_entropy,
                    'baseline_peak': base_peak,
                    'daga_peak': daga_peak,
                    'peak_change': daga_peak - base_peak,
                    'baseline_gini': base_gini,
                    'daga_gini': daga_gini,
                    'gini_change': daga_gini - base_gini,
                    'l2_distance': l2_dist,
                    'cosine_similarity': cos_sim
                })
    
    # Aggregate statistics by layer
    print("\n" + "-" * 70)
    print("Attention Statistics by Layer (Corrected Samples Only)")
    print("-" * 70)
    print(f"{'Layer':<10} {'Entropy Δ':>12} {'Peak Δ':>12} {'Gini Δ':>12} {'L2 Dist':>12} {'Cos Sim':>12}")
    print("-" * 70)
    
    for layer_idx in target_layers:
        layer_stats = [s for s in attention_stats if s['layer'] == layer_idx]
        if layer_stats:
            avg_entropy = np.mean([s['entropy_change'] for s in layer_stats])
            avg_peak = np.mean([s['peak_change'] for s in layer_stats])
            avg_gini = np.mean([s['gini_change'] for s in layer_stats])
            avg_l2 = np.mean([s['l2_distance'] for s in layer_stats])
            avg_cos = np.mean([s['cosine_similarity'] for s in layer_stats])
            
            print(f"Layer {layer_idx+1:<4} {avg_entropy:>+12.4f} {avg_peak:>+12.4f} {avg_gini:>+12.4f} {avg_l2:>12.4f} {avg_cos:>12.4f}")
    
    # Overall statistics
    print("-" * 70)
    if attention_stats:
        overall_entropy = np.mean([s['entropy_change'] for s in attention_stats])
        overall_peak = np.mean([s['peak_change'] for s in attention_stats])
        overall_gini = np.mean([s['gini_change'] for s in attention_stats])
        overall_l2 = np.mean([s['l2_distance'] for s in attention_stats])
        overall_cos = np.mean([s['cosine_similarity'] for s in attention_stats])
        
        print(f"{'Overall':<10} {overall_entropy:>+12.4f} {overall_peak:>+12.4f} {overall_gini:>+12.4f} {overall_l2:>12.4f} {overall_cos:>12.4f}")
    
    print("\nInterpretation:")
    print("  - Entropy Δ < 0: DAGA makes attention more focused")
    print("  - Peak Δ > 0: DAGA increases attention peak value")  
    print("  - Gini Δ > 0: DAGA makes attention more concentrated")
    print("  - L2 Distance: Magnitude of attention change")
    print("  - Cosine Sim < 1: Attention pattern changed direction")
    
    # Save quantitative results to CSV
    if attention_stats:
        import csv
        csv_path = output_dir / "attention_quantitative_analysis.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=attention_stats[0].keys())
            writer.writeheader()
            writer.writerows(attention_stats)
        print(f"\n✓ Quantitative analysis saved to: {csv_path}")
        
        # Create visualization of quantitative analysis
        create_quantitative_visualizations(attention_stats, target_layers, output_dir)
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    baseline_acc = sum(1 for x in improvements if x['baseline_correct']) / len(improvements)
    daga_acc = sum(1 for x in improvements if x['daga_correct']) / len(improvements)
    corrected = sum(1 for x in improvements if x['daga_correct'] and not x['baseline_correct'])
    
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"DAGA Accuracy: {daga_acc:.2%}")
    print(f"Samples corrected by DAGA: {corrected}")
    print(f"\nResults saved to: {output_dir}")


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DTD Multi-Layer Attention Visualization")
    
    parser.add_argument("--baseline_checkpoint", type=str, required=True,
                       help="Path to baseline model checkpoint")
    parser.add_argument("--daga_checkpoint", type=str, required=True,
                       help="Path to DAGA model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to DTD dataset")
    parser.add_argument("--output_dir", type=str, default="./visualization/results/dtd_multilayer",
                       help="Output directory for visualizations")
    parser.add_argument("--input_size", type=int, default=518,
                       help="Input image size")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to visualize")
    parser.add_argument("--max_eval", type=int, default=500,
                       help="Maximum samples to evaluate for finding improvements")
    parser.add_argument("--target_layers", type=int, nargs="+", default=None,
                       help="Specific layers to visualize (default: auto-detect)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dtd_visualization(args)

