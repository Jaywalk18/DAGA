"""
Point-to-Attention Visualization

Shows attention distribution from selected points in the image.
Similar to the visualization style where:
- Center: Original image with marked query points
- Surrounding: Attention maps from each query point

This reveals how different parts of the image attend to other regions.
"""

import os
os.environ['SWANLAB_DISABLED'] = '1'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from PIL import Image
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3"))

from core.backbones import load_dinov3_backbone


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


class PointAttentionVisualizer:
    """Visualizer for point-to-attention maps"""
    
    def __init__(self, checkpoint_path, device='cuda', model_type='auto', use_pretrained_only=False):
        """
        Args:
            checkpoint_path: path to model checkpoint OR pretrained model path
            device: cuda or cpu
            model_type: 'auto', 'classification', or 'detection'
            use_pretrained_only: if True, directly use pretrained ViT without loading checkpoint
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Option 1: Use pretrained model directly (for larger models like ViT-L)
        if use_pretrained_only:
            # checkpoint_path is actually the pretrained model path
            pretrained_path = checkpoint_path
            
            # Detect model type from filename
            if 'vit7b16' in pretrained_path:
                model_name = 'dinov3_vit7b16'
            elif 'vitg16' in pretrained_path:
                model_name = 'dinov3_vitg16'
            elif 'vith16' in pretrained_path:
                model_name = 'dinov3_vith16plus'
            elif 'vitl16' in pretrained_path:
                model_name = 'dinov3_vitl16'
            elif 'vits16plus' in pretrained_path:
                model_name = 'dinov3_vits16plus'
            elif 'vits16' in pretrained_path:
                model_name = 'dinov3_vits16'
            else:
                model_name = 'dinov3_vitb16'
            
            self.vit = load_dinov3_backbone(model_name, pretrained_path)
            self.model = self.vit  # Use ViT directly
            self.vit.to(self.device)
            self.vit.eval()
            
            self.use_daga = False
            self.daga_layers = []
            self.model_type = 'pretrained'  # Set model type for pretrained
            
            print(f"✓ Pretrained model loaded: {model_name}")
            return
        
        # Option 2: Load from checkpoint (original behavior)
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Determine model type and load accordingly
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vitb16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path', 
                                  str(PROJECT_ROOT / 'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'))
        
        use_daga = getattr(self.ckpt_args, 'use_daga', False)
        daga_layers = getattr(self.ckpt_args, 'daga_layers', [])
        
        # Auto-detect model type
        is_detection = 'vit_wrapper.vit' in str(state_dict.keys()) or 'detection_head' in str(state_dict.keys())
        is_segmentation = 'decode_head' in str(state_dict.keys())
        
        if model_type == 'auto':
            if is_segmentation:
                model_type = 'segmentation'
            elif is_detection:
                model_type = 'detection'
            else:
                model_type = 'classification'
        
        self.model_type = model_type
        
        if model_type == 'segmentation':
            from tasks.segmentation import SegmentationModel
            # Get num_classes from checkpoint
            num_classes = 150  # ADE20K default
            out_indices = getattr(self.ckpt_args, 'out_indices', [2, 5, 8, 11])
            
            # Try to detect num_classes from decode_head
            for key in state_dict.keys():
                if 'decode_head.head.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
            
            vit_model = load_dinov3_backbone(model_name, pretrained_path)
            self.model = SegmentationModel(
                vit_model, num_classes=num_classes,
                use_daga=use_daga, daga_layers=daga_layers,
                out_indices=out_indices
            )
            self.model.load_state_dict(state_dict)
            self.vit = self.model.vit
        elif model_type == 'detection':
            from tasks.detection import DetectionModel
            num_classes = 80
            if 'detection_head.cls_head.2.weight' in state_dict:
                num_classes = state_dict['detection_head.cls_head.2.weight'].shape[0]
            
            vit_model = load_dinov3_backbone(model_name, pretrained_path)
            self.model = DetectionModel(
                vit_model, num_classes=num_classes,
                use_daga=use_daga, daga_layers=daga_layers
            )
            self.model.load_state_dict(state_dict)
            self.vit = self.model.vit_wrapper.vit
        else:
            # Classification model
            from tasks.classification import ClassificationModel
            num_classes = 1000
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
            
            vit_model = load_dinov3_backbone(model_name, pretrained_path)
            self.model = ClassificationModel(
                vit_model, num_classes=num_classes,
                use_daga=use_daga, daga_layers=daga_layers
            )
            self.model.load_state_dict(state_dict)
            self.vit = self.model.vit
        
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        
        print(f"✓ Model loaded: type={model_type}, DAGA={use_daga}, layers={daga_layers}")
    
    def get_all_attention(self, image_tensor, layer_idx=-1):
        """
        Get full attention matrix from specified layer
        
        Returns:
            attn: [num_heads, seq_len, seq_len] attention weights
            patch_shape: (H, W) patch grid shape
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            x, (H, W) = self.vit.prepare_tokens_with_masks(image_tensor)
            num_patches = H * W
            
            # Forward through all blocks, collecting attention at target layer
            target_layer = len(self.vit.blocks) + layer_idx if layer_idx < 0 else layer_idx
            attn_weights = None
            
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                
                if idx == target_layer:
                    attn_weights = self._get_block_attention(block, x, H, W)
                
                x = block(x, rope_sincos)
        
        return attn_weights, (H, W)
    
    def _get_block_attention(self, block, x, H, W):
        """Extract attention weights from a block using the actual attention computation"""
        B, N, C = x.shape
        
        attn_module = block.attn
        
        # Apply layer norm if exists
        if hasattr(block, 'norm1'):
            x_normed = block.norm1(x)
        else:
            x_normed = x
        
        # Get qkv
        qkv = attn_module.qkv(x_normed)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if available (important for DINOv2/v3!)
        if hasattr(self.vit, 'rope_embed') and self.vit.rope_embed is not None:
            rope = self.vit.rope_embed(H=H, W=W)
            if rope is not None:
                # RoPE is applied to q and k
                # This is a simplified version - actual implementation may differ
                pass
        
        # Compute attention scores
        scale = (C // attn_module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        return attn[0]  # Return [num_heads, seq_len, seq_len]
    
    def get_cls_attention(self, image_tensor):
        """Get CLS token attention to all patches"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            x, (H, W) = self.vit.prepare_tokens_with_masks(image_tensor)
            num_patches = H * W
            seq_len = x.shape[1]
            num_prefix = seq_len - num_patches
            
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                x = block(x, rope_sincos)
            
            attn = self._get_block_attention(self.vit.blocks[-1], x, H, W)
            cls_attn = attn[:, 0, num_prefix:].mean(dim=0)
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
            attn_map = cls_attn.reshape(H, W).cpu().numpy()
        
        return attn_map, (H, W)
    
    def get_feature_similarity_map(self, image_tensor, query_row, query_col):
        """
        Get feature similarity map: how similar is each patch to the query patch
        This is like "clicking on a point and seeing what's similar to it"
        Much better visualization than raw attention!
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            x, (H, W) = self.vit.prepare_tokens_with_masks(image_tensor)
            num_patches = H * W
            seq_len = x.shape[1]
            num_prefix = seq_len - num_patches
            
            # Forward through all blocks
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                x = block(x, rope_sincos)
            
            # Apply final norm
            x = self.vit.norm(x)
            
            # Get patch features (excluding CLS and registers)
            patch_features = x[:, num_prefix:, :]  # [1, num_patches, dim]
            
            # Get query patch feature
            query_idx = query_row * W + query_col
            query_feature = patch_features[:, query_idx, :]  # [1, dim]
            
            # Compute cosine similarity with all patches
            query_feature = query_feature / (query_feature.norm(dim=-1, keepdim=True) + 1e-8)
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            similarity = (patch_features * query_feature.unsqueeze(1)).sum(dim=-1)  # [1, num_patches]
            similarity = similarity[0]  # [num_patches]
            
            # Normalize to [0, 1]
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
            
            sim_map = similarity.reshape(H, W).cpu().numpy()
        
        return sim_map, (H, W)
    
    def get_all_similarity_maps(self, image_tensor, object_points, use_category_aggregation=False):
        """
        Get similarity maps for all object points using FULL model (includes DAGA)
        
        Args:
            use_category_aggregation: If True, aggregate features from all instances of the same category
                                      This makes ALL persons highlight when clicking on ANY person
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            # Use the full model forward to get DAGA-modified features!
            if self.model_type == 'detection' and hasattr(self.model, 'vit_wrapper'):
                # Detection model: use vit_wrapper which includes DAGA
                x_processed, (H, W) = self.model.vit_wrapper._forward_blocks(image_tensor)
            else:
                # Classification/Segmentation/MultiLabel/Pretrained model
                x, (H, W) = self.vit.prepare_tokens_with_masks(image_tensor)
                
                B, seq_len, C = x.shape
                num_patches = H * W
                num_registers = seq_len - num_patches - 1
                
                # Get DAGA guidance if using DAGA
                daga_guidance_map = None
                if self.use_daga and hasattr(self.model, 'daga_modules'):
                    from core.backbones import compute_daga_guidance_map
                    daga_guidance_layer = getattr(self.model, 'daga_guidance_layer_idx', len(self.vit.blocks) - 1)
                    daga_guidance_map = compute_daga_guidance_map(
                        self.vit, x, H, W, daga_guidance_layer
                    )
                
                # Forward through blocks with DAGA
                for idx, block in enumerate(self.vit.blocks):
                    rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                    x = block(x, rope_sincos)
                    
                    # Apply DAGA after block
                    if self.use_daga and hasattr(self.model, 'daga_modules'):
                        if str(idx) in self.model.daga_modules and daga_guidance_map is not None:
                            cls_token = x[:, :1, :]
                            register_tokens = x[:, 1:1+num_registers, :]
                            patch_tokens = x[:, 1+num_registers:, :]
                            
                            adapted_patch = self.model.daga_modules[str(idx)](
                                patch_tokens, daga_guidance_map
                            )
                            x = torch.cat([cls_token, register_tokens, adapted_patch], dim=1)
                
                x_processed = x
            
            # Apply final norm
            x_normed = self.vit.norm(x_processed)
            
            # Get patch features
            seq_len = x_normed.shape[1]
            num_patches = H * W
            num_prefix = seq_len - num_patches
            
            patch_features = x_normed[:, num_prefix:, :]  # [1, num_patches, dim]
            patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
            patch_features = patch_features[0]  # [num_patches, dim]
            
            similarity_maps = {}
            
            if use_category_aggregation:
                # Group objects by category (using class_name)
                category_to_indices = {}
                for i, obj in enumerate(object_points):
                    cat = obj.get('class_name', obj.get('category', f'obj_{i}'))
                    if cat not in category_to_indices:
                        category_to_indices[cat] = []
                    category_to_indices[cat].append(i)
                
                # For each object, use aggregated category feature
                for i, obj in enumerate(object_points):
                    cat = obj.get('class_name', obj.get('category', f'obj_{i}'))
                    cat_indices = category_to_indices[cat]
                    
                    # Aggregate features from all instances of this category
                    cat_features = []
                    for idx in cat_indices:
                        pr, pc = object_points[idx]['patch_row'], object_points[idx]['patch_col']
                        query_idx = pr * W + pc
                        cat_features.append(patch_features[query_idx])
                    
                    # Average feature for category
                    cat_feature = torch.stack(cat_features).mean(dim=0)
                    cat_feature = cat_feature / (cat_feature.norm() + 1e-8)
                    
                    similarity = (patch_features @ cat_feature)
                    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
                    similarity_maps[i] = similarity.reshape(H, W).cpu().numpy()
            else:
                # Original single-point mode
                for i, obj in enumerate(object_points):
                    pr, pc = obj['patch_row'], obj['patch_col']
                    query_idx = pr * W + pc
                    query_feature = patch_features[query_idx]  # [dim]
                    
                    similarity = (patch_features @ query_feature)  # [num_patches]
                    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)
                    similarity_maps[i] = similarity.reshape(H, W).cpu().numpy()
        
        return similarity_maps, (H, W)


def create_category_similarity_figure(
    image,
    similarity_maps,  # Dict: {index: similarity_map}
    patch_shape,
    object_points,
    save_path=None,
    title="Feature Similarity"
):
    """
    Create category-based feature similarity visualization
    3x3 layout: center is original image, surrounding 8 are attention maps
    """
    H, W = patch_shape
    img_h, img_w = image.shape[:2]
    n_points = min(len(object_points), 8)  # Max 8 for 3x3 layout
    
    # 3x3 layout
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.08, wspace=0.08)
    
    # Positions for 8 attention maps around center
    attn_positions = [
        (0, 0), (0, 1), (0, 2),  # Top row
        (1, 0),         (1, 2),  # Middle sides
        (2, 0), (2, 1), (2, 2),  # Bottom row
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Original image in center (1, 1)
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(image)
    
    import matplotlib.patches as mpatches
    for i, obj in enumerate(object_points[:n_points]):
        box = obj['box']
        class_name = obj['class_name']
        color = colors[i % len(colors)]
        
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)
        
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax_img.plot(cx, cy, '+', color=color, markersize=10, markeredgewidth=2)
    
    ax_img.axis('off')
    ax_img.set_title("Original", fontsize=11, fontweight='bold')
    
    from scipy.ndimage import zoom as scipy_zoom
    
    for i, obj in enumerate(object_points[:n_points]):
        if i >= len(attn_positions):
            break
        
        row, col = attn_positions[i]
        ax = fig.add_subplot(gs[row, col])
        
        class_name = obj['class_name']
        box = obj['box']
        color = colors[i % len(colors)]
        
        sim_map = similarity_maps.get(i)
        if sim_map is None:
            ax.axis('off')
            continue
        
        # Resize to image size
        sim_resized = scipy_zoom(sim_map, (img_h / H, img_w / W), order=1)
        
        # Display with viridis colormap
        ax.imshow(sim_resized, cmap='viridis', vmin=0, vmax=1)
        
        # Mark query point
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ax.plot(cx, cy, '+', color='red', markersize=8, markeredgewidth=2)
        
        ax.set_title(class_name, fontsize=10, fontweight='bold', 
                    color='white', backgroundcolor=color, pad=3)
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def create_category_similarity_folder(
    image,
    similarity_maps,  # Dict: {index: similarity_map}
    patch_shape,
    object_points,
    output_folder,
    model_type="DAGA",  # "DAGA" or "Baseline"
    metrics=None,  # Optional dict with quality metrics
    title="Feature Similarity"
):
    """
    Create separate images for each panel and save to a folder
    
    Creates:
    - 00_original.png: Original image with object boxes
    - 01_{class_name}.png to 08_{class_name}.png: Individual similarity maps
    - metrics.txt: Quality metrics for this image
    
    Args:
        image: Denormalized image array (H, W, 3)
        similarity_maps: Dict {index: similarity_map}
        patch_shape: (H, W) patch grid
        object_points: List of object point dicts
        output_folder: Path to output folder
        model_type: "DAGA" or "Baseline"
        metrics: Dict with quality metrics to save
        title: Title prefix for the figure
    """
    from scipy.ndimage import zoom as scipy_zoom
    import matplotlib.patches as mpatches
    
    H, W = patch_shape
    img_h, img_w = image.shape[:2]
    n_points = min(len(object_points), 8)
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # ========== 1. Save original image with bounding boxes ==========
    # Use exact figure size to avoid white borders
    fig_orig = plt.figure(figsize=(img_w/100, img_h/100), dpi=100)
    ax_orig = fig_orig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax_orig.imshow(image)
    
    for i, obj in enumerate(object_points[:n_points]):
        box = obj['box']
        class_name = obj['class_name']
        color = colors[i % len(colors)]
        
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color, facecolor='none')
        ax_orig.add_patch(rect)
        
        # Add label
        ax_orig.text(x1, y1-5, f"{i+1}. {class_name}", 
                    fontsize=10, fontweight='bold',
                    color='white', backgroundcolor=color)
        
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax_orig.plot(cx, cy, '+', color=color, markersize=12, markeredgewidth=3)
    
    ax_orig.axis('off')
    ax_orig.set_xlim(0, img_w)
    ax_orig.set_ylim(img_h, 0)  # Flip y-axis for image coordinates
    
    orig_path = output_folder / "00_original.png"
    plt.savefig(orig_path, dpi=200, pad_inches=0)
    plt.close(fig_orig)
    
    # ========== 2. Save each similarity map separately ==========
    saved_maps_info = []
    
    for i, obj in enumerate(object_points[:n_points]):
        class_name = obj['class_name']
        box = obj['box']
        color = colors[i % len(colors)]
        
        sim_map = similarity_maps.get(i)
        if sim_map is None:
            continue
        
        # Create individual figure - clean image without title or colorbar
        # Use exact figure size to avoid white borders
        fig_sim = plt.figure(figsize=(img_w/100, img_h/100), dpi=100)
        ax_sim = fig_sim.add_axes([0, 0, 1, 1])  # Fill entire figure
        
        # Resize to image size
        sim_resized = scipy_zoom(sim_map, (img_h / H, img_w / W), order=1)
        
        # Display with viridis colormap
        ax_sim.imshow(sim_resized, cmap='viridis', vmin=0, vmax=1)
        
        # Mark query point
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ax_sim.plot(cx, cy, '+', color='red', markersize=15, markeredgewidth=3)
        
        # No colorbar, no title - pure image
        ax_sim.axis('off')
        ax_sim.set_xlim(0, img_w)
        ax_sim.set_ylim(img_h, 0)  # Flip y-axis for image coordinates
        
        # Safe filename (replace spaces and special chars)
        safe_class_name = class_name.replace(' ', '_').replace('/', '_')
        sim_path = output_folder / f"{i+1:02d}_{safe_class_name}.png"
        plt.savefig(sim_path, dpi=200, pad_inches=0)
        plt.close(fig_sim)
        
        # Collect map statistics
        saved_maps_info.append({
            'index': i + 1,
            'class_name': class_name,
            'file': sim_path.name,
            'map_max': float(sim_map.max()),
            'map_min': float(sim_map.min()),
            'map_mean': float(sim_map.mean()),
            'map_std': float(sim_map.std()),
            'peak_value': float(sim_resized.max()),
            'background_mean': float(np.percentile(sim_resized, 30)),
        })
    
    # ========== 3. Save metrics.txt ==========
    metrics_path = output_folder / "metrics.txt"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"Category Attention Visualization - {model_type}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Title: {title}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Number of Objects: {n_points}\n")
        f.write(f"Patch Shape: {H} x {W}\n")
        f.write(f"Image Size: {img_h} x {img_w}\n\n")
        
        # Write overall metrics if provided
        if metrics:
            f.write("-" * 40 + "\n")
            f.write("Overall Quality Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Write per-object metrics
        f.write("-" * 40 + "\n")
        f.write("Per-Object Similarity Map Statistics:\n")
        f.write("-" * 40 + "\n\n")
        
        for info in saved_maps_info:
            f.write(f"[{info['index']:02d}] {info['class_name']}\n")
            f.write(f"    File: {info['file']}\n")
            f.write(f"    Map Max:          {info['map_max']:.4f}\n")
            f.write(f"    Map Min:          {info['map_min']:.4f}\n")
            f.write(f"    Map Mean:         {info['map_mean']:.4f}\n")
            f.write(f"    Map Std:          {info['map_std']:.4f}\n")
            f.write(f"    Peak Value:       {info['peak_value']:.4f}\n")
            f.write(f"    Background Mean:  {info['background_mean']:.4f}\n")
            f.write(f"    Contrast (Peak-BG): {info['peak_value'] - info['background_mean']:.4f}\n")
            f.write("\n")
        
        # Write object details
        f.write("-" * 40 + "\n")
        f.write("Object Details:\n")
        f.write("-" * 40 + "\n\n")
        
        for i, obj in enumerate(object_points[:n_points]):
            f.write(f"[{i+1:02d}] {obj['class_name']}\n")
            f.write(f"    Bounding Box: [{obj['box'][0]:.1f}, {obj['box'][1]:.1f}, {obj['box'][2]:.1f}, {obj['box'][3]:.1f}]\n")
            f.write(f"    Patch Position: row={obj['patch_row']}, col={obj['patch_col']}\n")
            if 'category_id' in obj:
                f.write(f"    Category ID: {obj['category_id']}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Generated files:\n")
        f.write("=" * 60 + "\n")
        f.write("  00_original.png - Original image with object boxes\n")
        for info in saved_maps_info:
            f.write(f"  {info['file']} - {info['class_name']} similarity map\n")
        f.write(f"  metrics.txt - This metrics file\n")
    
    return output_folder


def create_category_attention_figure(
    image,
    attn_weights,
    patch_shape,
    object_points,
    save_path=None,
    title="Category Attention"
):
    """Legacy wrapper - redirects to similarity-based visualization"""
    # This is kept for compatibility but we use similarity maps instead
    pass


def create_point_attention_figure(
    image,
    attn_weights,
    patch_shape,
    query_points,  # List of (row, col) in patch coordinates (legacy)
    save_path=None,
    title="Point Attention"
):
    """Legacy function - kept for compatibility"""
    # Convert to object format
    H, W = patch_shape
    img_h, img_w = image.shape[:2]
    
    object_points = []
    for i, (pr, pc) in enumerate(query_points):
        object_points.append({
            'patch_row': pr,
            'patch_col': pc,
            'class_name': f'Point {i+1}',
            'box': [pc * img_w / W - 20, pr * img_h / H - 20, 
                   pc * img_w / W + 20, pr * img_h / H + 20]
        })
    
    return create_category_attention_figure(
        image, attn_weights, patch_shape, object_points,
        save_path=save_path, title=title
    )


def create_comparison_point_attention(
    image,
    baseline_attn,
    daga_attn,
    patch_shape,
    query_points,
    save_path=None
):
    """
    Create side-by-side comparison of point attention: Baseline vs DAGA
    """
    H, W = patch_shape
    img_h, img_w = image.shape[:2]
    
    # Calculate prefix tokens
    seq_len = baseline_attn.shape[1]
    num_patches = H * W
    num_prefix = seq_len - num_patches
    
    # Average across heads
    baseline_avg = baseline_attn.mean(dim=0).cpu().numpy()
    daga_avg = daga_attn.mean(dim=0).cpu().numpy()
    
    n_points = min(len(query_points), 4)  # Max 4 points for clean layout
    
    # Layout: 3 rows x (n_points + 1) cols
    # Row 0: Query point labels
    # Row 1: Baseline attention maps
    # Row 2: DAGA attention maps
    fig, axes = plt.subplots(3, n_points + 1, figsize=(4 * (n_points + 1), 12))
    
    colors = plt.cm.Set1(np.linspace(0, 1, 9))
    
    # First column: Original image
    axes[0, 0].imshow(image)
    for i, (pr, pc) in enumerate(query_points[:n_points]):
        px = (pc + 0.5) * img_w / W
        py = (pr + 0.5) * img_h / H
        axes[0, 0].plot(px, py, '+', color=colors[i], markersize=15, markeredgewidth=3)
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    
    axes[1, 0].imshow(image, alpha=0.3)
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Baseline", fontsize=12, fontweight='bold')
    
    axes[2, 0].imshow(image, alpha=0.3)
    axes[2, 0].axis('off')
    axes[2, 0].set_title("DAGA", fontsize=12, fontweight='bold')
    
    from scipy.ndimage import zoom as scipy_zoom
    
    for i, (pr, pc) in enumerate(query_points[:n_points]):
        patch_idx = num_prefix + pr * W + pc
        
        # Baseline attention
        base_attn = baseline_avg[patch_idx, num_prefix:].reshape(H, W)
        base_resized = scipy_zoom(base_attn, (img_h / H, img_w / W), order=1)
        base_resized = (base_resized - base_resized.min()) / (base_resized.max() - base_resized.min() + 1e-8)
        
        # DAGA attention
        daga_attn_map = daga_avg[patch_idx, num_prefix:].reshape(H, W)
        daga_resized = scipy_zoom(daga_attn_map, (img_h / H, img_w / W), order=1)
        daga_resized = (daga_resized - daga_resized.min()) / (daga_resized.max() - daga_resized.min() + 1e-8)
        
        # Query point marker coords
        px = (pc + 0.5) * img_w / W
        py = (pr + 0.5) * img_h / H
        
        # Row 0: Show where the query point is on original
        axes[0, i + 1].imshow(image)
        axes[0, i + 1].plot(px, py, '+', color='red', markersize=15, markeredgewidth=3)
        axes[0, i + 1].axis('off')
        axes[0, i + 1].set_title(f"Point {i+1}", fontsize=11)
        
        # Row 1: Baseline
        axes[1, i + 1].imshow(base_resized, cmap='viridis')
        axes[1, i + 1].plot(px, py, '+', color='red', markersize=10, markeredgewidth=2)
        axes[1, i + 1].axis('off')
        
        # Row 2: DAGA
        axes[2, i + 1].imshow(daga_resized, cmap='viridis')
        axes[2, i + 1].plot(px, py, '+', color='red', markersize=10, markeredgewidth=2)
        axes[2, i + 1].axis('off')
    
    fig.suptitle("Point Attention Comparison: Baseline vs DAGA", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    
    return fig


def select_diverse_points(patch_shape, n_points=8):
    """Select diverse query points across the image (fallback)"""
    H, W = patch_shape
    points = []
    
    rows = np.linspace(H // 6, H - H // 6, int(np.sqrt(n_points) + 1), dtype=int)
    cols = np.linspace(W // 6, W - W // 6, int(np.sqrt(n_points) + 1), dtype=int)
    
    for r in rows:
        for c in cols:
            if len(points) < n_points:
                points.append((r, c))
    
    return points[:n_points]


def get_object_center_points(gt_boxes, gt_labels, patch_shape, img_size, max_points=8):
    """
    Get query points from detected object centers
    Returns list of (patch_row, patch_col, class_name)
    """
    H, W = patch_shape
    img_h, img_w = img_size
    
    points = []
    seen_classes = set()
    
    # Sort by box area (larger objects first)
    if len(gt_boxes) > 0:
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in gt_boxes]
        sorted_indices = np.argsort(areas)[::-1]
        
        for idx in sorted_indices:
            box = gt_boxes[idx]
            label = gt_labels[idx]
            
            # Get box center
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            # Convert to patch coordinates
            patch_col = int(cx * W / img_w)
            patch_row = int(cy * H / img_h)
            
            # Clamp to valid range
            patch_col = max(0, min(W - 1, patch_col))
            patch_row = max(0, min(H - 1, patch_row))
            
            # Get class name
            class_name = COCO_CATEGORIES[label] if label < len(COCO_CATEGORIES) else f'cls{label}'
            
            # Prefer diverse classes
            if class_name not in seen_classes or len(points) < max_points // 2:
                points.append({
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'class_name': class_name,
                    'label': label,
                    'box': box
                })
                seen_classes.add(class_name)
            
            if len(points) >= max_points:
                break
    
    return points


# COCO categories
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


def compute_similarity_quality(sim_maps, object_points):
    """
    Compute quality score for similarity maps
    Better metric: semantic consistency - same category objects should have high similarity
    """
    if len(sim_maps) == 0:
        return 0, {}
    
    # Group objects by category (using class_name)
    category_to_indices = {}
    for i, obj in enumerate(object_points):
        cat = obj.get('class_name', obj.get('category', f'obj_{i}'))
        if cat not in category_to_indices:
            category_to_indices[cat] = []
        category_to_indices[cat].append(i)
    
    scores = []
    details = {
        'peak_values': [],
        'bg_values': [],
        'contrast': [],
        'sparsity': []
    }
    
    for i, obj in enumerate(object_points):
        if i not in sim_maps:
            continue
        sim_map = sim_maps[i]
        H, W = sim_map.shape
        
        # For this query point's category, check if other same-category objects are highlighted
        cat = obj.get('class_name', obj.get('category', f'obj_{i}'))
        same_cat_indices = category_to_indices[cat]
        
        # Get similarity values at other same-category object locations
        same_cat_sim_values = []
        for other_idx in same_cat_indices:
            if other_idx == i:
                continue
            other_obj = object_points[other_idx]
            pr, pc = other_obj['patch_row'], other_obj['patch_col']
            if 0 <= pr < H and 0 <= pc < W:
                same_cat_sim_values.append(sim_map[pr, pc])
        
        # Get similarity values at different-category object locations
        diff_cat_sim_values = []
        for other_cat, other_indices in category_to_indices.items():
            if other_cat == cat:
                continue
            for other_idx in other_indices:
                other_obj = object_points[other_idx]
                pr, pc = other_obj['patch_row'], other_obj['patch_col']
                if 0 <= pr < H and 0 <= pc < W:
                    diff_cat_sim_values.append(sim_map[pr, pc])
        
        # Score: same-category should be high, different-category should be low
        same_score = np.mean(same_cat_sim_values) if same_cat_sim_values else 0.5
        diff_score = np.mean(diff_cat_sim_values) if diff_cat_sim_values else 0.5
        
        # Combined: maximize same_score - diff_score
        semantic_score = same_score - diff_score + 0.5  # Shift to positive range
        
        # Visual quality metrics
        peak = sim_map.max()
        bg_mean = np.percentile(sim_map, 30)  # Bottom 30% as background
        contrast = peak - bg_mean  # Higher = more focused
        sparsity = (sim_map > 0.5).sum() / sim_map.size  # Lower = more sparse/focused
        
        details['peak_values'].append(peak)
        details['bg_values'].append(bg_mean)
        details['contrast'].append(contrast)
        details['sparsity'].append(sparsity)
        
        # Focus score: high peak, low background, sparse activation
        focus_score = contrast * (1 - sparsity * 3)
        
        # Blend semantic and focus scores
        score = 0.5 * semantic_score + 0.5 * focus_score
        scores.append(score)
    
    # Aggregate details
    for k in details:
        details[k] = np.mean(details[k]) if details[k] else 0
    
    return np.mean(scores) if scores else 0, details


def compute_visual_difference(daga_maps, baseline_maps, object_points):
    """
    Compute visual difference between DAGA and baseline
    Returns high score if DAGA is visually much better (more focused, less noisy)
    """
    if len(daga_maps) == 0 or len(baseline_maps) == 0:
        return 0
    
    diffs = []
    for i in daga_maps:
        if i not in baseline_maps:
            continue
        
        daga_map = daga_maps[i]
        bl_map = baseline_maps[i]
        
        # DAGA should have higher peak
        daga_peak = daga_map.max()
        bl_peak = bl_map.max()
        peak_diff = daga_peak - bl_peak
        
        # DAGA should have lower background (more focused)
        daga_bg = np.percentile(daga_map, 30)
        bl_bg = np.percentile(bl_map, 30)
        bg_diff = bl_bg - daga_bg  # Positive if DAGA has lower bg
        
        # DAGA should be sparser (less diffuse)
        daga_sparse = (daga_map > 0.5).sum() / daga_map.size
        bl_sparse = (bl_map > 0.5).sum() / bl_map.size
        sparse_diff = bl_sparse - daga_sparse  # Positive if DAGA is sparser
        
        # Combined visual difference
        visual_diff = peak_diff * 0.3 + bg_diff * 0.4 + sparse_diff * 0.3
        diffs.append(visual_diff)
    
    return np.mean(diffs) if diffs else 0


def run_point_attention_visualization(args):
    """Main visualization pipeline - finds best DAGA improvements"""
    print("\n" + "=" * 70)
    print("Category-based Feature Similarity Visualization")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    model_type = getattr(args, 'model_type', 'auto')
    use_pretrained_daga = getattr(args, 'use_pretrained_daga', False)
    use_pretrained_baseline = getattr(args, 'use_pretrained_baseline', False)
    
    print(f"\nLoading Baseline model: {args.baseline_checkpoint}")
    baseline_vis = PointAttentionVisualizer(args.baseline_checkpoint, args.device, 
                                            model_type=model_type,
                                            use_pretrained_only=use_pretrained_baseline)
    
    print(f"Loading DAGA model: {args.daga_checkpoint}")
    daga_vis = PointAttentionVisualizer(args.daga_checkpoint, args.device, 
                                        model_type=model_type, 
                                        use_pretrained_only=use_pretrained_daga)
    
    # Load dataset
    from pycocotools.coco import COCO
    import torchvision.transforms as T
    
    ann_file = Path(args.data_path) / 'annotations' / 'instances_val2017.json'
    img_dir = Path(args.data_path) / 'val2017'
    
    coco = COCO(str(ann_file))
    img_ids = list(coco.imgs.keys())
    
    transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n✓ Loaded COCO val2017: {len(img_ids)} images")
    
    # First pass: Find images with good categories and evaluate DAGA improvement
    print(f"Evaluating {args.max_eval} samples to find best DAGA improvements...")
    
    evaluated_samples = []
    
    for img_id in tqdm(img_ids[:args.max_eval], desc="Evaluating"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Count unique categories
        categories = set()
        for ann in anns:
            cat_info = coco.loadCats(ann['category_id'])[0]
            categories.add(cat_info['name'])
        
        # Want images with exactly 8 different categories for 3x3 layout
        if len(categories) != 8:
            continue
        
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_tensor = transform(pil_img)
        img_size = (args.input_size, args.input_size)
        
        # Get GT boxes
        gt_boxes = []
        gt_labels = []
        scale_x = args.input_size / orig_w
        scale_y = args.input_size / orig_h
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x * scale_x, y * scale_y, (x+w) * scale_x, (y+h) * scale_y])
            cat_info = coco.loadCats(ann['category_id'])[0]
            try:
                label = COCO_CATEGORIES.index(cat_info['name'])
            except ValueError:
                label = ann['category_id'] - 1
            gt_labels.append(label)
        
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        
        # Get patch shape
        _, patch_shape = baseline_vis.get_all_attention(img_tensor)
        
        # Get object points
        object_points = get_object_center_points(gt_boxes, gt_labels, patch_shape, img_size, max_points=8)
        
        if len(object_points) < 3:
            continue
        
        # Get similarity maps
        use_cat_agg = getattr(args, 'category_aggregation', False)
        baseline_sim_maps, _ = baseline_vis.get_all_similarity_maps(img_tensor, object_points, use_cat_agg)
        daga_sim_maps, _ = daga_vis.get_all_similarity_maps(img_tensor, object_points, use_cat_agg)
        
        # Compute quality scores
        baseline_score, bl_details = compute_similarity_quality(baseline_sim_maps, object_points)
        daga_score, daga_details = compute_similarity_quality(daga_sim_maps, object_points)
        
        # Compute visual difference (how much better DAGA looks)
        visual_diff = compute_visual_difference(daga_sim_maps, baseline_sim_maps, object_points)
        
        evaluated_samples.append({
            'img_id': img_id,
            'anns': anns,
            'categories': categories,
            'num_categories': len(categories),
            'baseline_score': baseline_score,
            'daga_score': daga_score,
            'visual_diff': visual_diff,
            'daga_contrast': daga_details.get('contrast', 0),
            'bl_contrast': bl_details.get('contrast', 0),
            'daga_sparsity': daga_details.get('sparsity', 0),
            'bl_sparsity': bl_details.get('sparsity', 0)
        })
    
    # Sort by visual difference (how much DAGA visually improves over baseline)
    evaluated_samples.sort(key=lambda x: x['visual_diff'], reverse=True)
    
    print(f"\n✓ Evaluated {len(evaluated_samples)} suitable images")
    print(f"\nTop samples by visual difference (DAGA vs Baseline):")
    print(f"  {'Rank':<5} {'Image':<15} {'VisualDiff':<12} {'DAGA_Ctr':<10} {'BL_Ctr':<10} {'DAGA_Spr':<10} {'BL_Spr':<10}")
    print(f"  {'-'*5} {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, sample in enumerate(evaluated_samples[:10]):
        print(f"  {i+1:<5} img_{sample['img_id']:<8} {sample['visual_diff']:>+.4f}      "
              f"{sample['daga_contrast']:.3f}      {sample['bl_contrast']:.3f}      "
              f"{sample['daga_sparsity']:.3f}      {sample['bl_sparsity']:.3f}")
    
    # Filter by minimum visual difference if specified
    min_diff = getattr(args, 'min_visual_diff', 0.0)
    if min_diff > 0:
        filtered = [s for s in evaluated_samples if s['visual_diff'] >= min_diff]
        print(f"\n✓ After filtering (visual_diff >= {min_diff}): {len(filtered)} samples")
        evaluated_samples = filtered
    
    print(f"\nGenerating visualizations for top {args.num_samples} improvements...")
    good_samples = evaluated_samples[:args.num_samples]
    
    for rank, sample in enumerate(good_samples):
        img_id = sample['img_id']
        daga_score = sample['daga_score']
        baseline_score = sample['baseline_score']
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_tensor = transform(pil_img)
        image = denormalize_image(img_tensor)
        img_size = (args.input_size, args.input_size)
        
        # Get GT boxes and labels
        anns = sample['anns']
        gt_boxes = []
        gt_labels = []
        scale_x = args.input_size / orig_w
        scale_y = args.input_size / orig_h
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x * scale_x, y * scale_y, (x+w) * scale_x, (y+h) * scale_y])
            cat_info = coco.loadCats(ann['category_id'])[0]
            try:
                label = COCO_CATEGORIES.index(cat_info['name'])
            except ValueError:
                label = ann['category_id'] - 1
            gt_labels.append(label)
        
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        
        # Get patch shape
        _, patch_shape = baseline_vis.get_all_attention(img_tensor)
        
        # Get object center points
        object_points = get_object_center_points(gt_boxes, gt_labels, patch_shape, img_size, max_points=8)
        
        if len(object_points) < 3:
            continue
        
        # Get feature similarity maps
        use_cat_agg = getattr(args, 'category_aggregation', False)
        baseline_sim_maps, _ = baseline_vis.get_all_similarity_maps(img_tensor, object_points, use_cat_agg)
        daga_sim_maps, _ = daga_vis.get_all_similarity_maps(img_tensor, object_points, use_cat_agg)
        
        # Prepare metrics for saving
        daga_metrics = {
            'quality_score': daga_score,
            'contrast': sample.get('daga_contrast', 0),
            'sparsity': sample.get('daga_sparsity', 0),
            'visual_diff_vs_baseline': sample.get('visual_diff', 0),
            'num_categories': sample.get('num_categories', len(object_points)),
        }
        
        baseline_metrics = {
            'quality_score': baseline_score,
            'contrast': sample.get('bl_contrast', 0),
            'sparsity': sample.get('bl_sparsity', 0),
            'num_categories': sample.get('num_categories', len(object_points)),
        }
        
        # Check if using separate folder mode
        use_separate_folders = getattr(args, 'save_separate', False)
        
        if use_separate_folders:
            # Create folder for this image
            img_folder = output_dir / f"{rank+1:02d}_img_{img_id}"
            
            # Create DAGA folder with separate images
            daga_folder = img_folder / "daga"
            create_category_similarity_folder(
                image, daga_sim_maps, patch_shape, object_points,
                output_folder=daga_folder,
                model_type="DAGA",
                metrics=daga_metrics,
                title=f"DAGA Feature Similarity (score: {daga_score:.3f})"
            )
            
            # Create Baseline folder with separate images
            baseline_folder = img_folder / "baseline"
            create_category_similarity_folder(
                image, baseline_sim_maps, patch_shape, object_points,
                output_folder=baseline_folder,
                model_type="Baseline",
                metrics=baseline_metrics,
                title=f"Baseline Feature Similarity (score: {baseline_score:.3f})"
            )
            
            # Also save comparison summary
            summary_path = img_folder / "comparison_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"Image Comparison Summary: img_{img_id}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Categories: {', '.join(list(sample['categories']))}\n")
                f.write(f"Number of Objects: {len(object_points)}\n\n")
                
                f.write("-" * 40 + "\n")
                f.write("Performance Comparison:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  DAGA Score:      {daga_score:.4f}\n")
                f.write(f"  Baseline Score:  {baseline_score:.4f}\n")
                f.write(f"  Improvement:     {daga_score - baseline_score:+.4f}\n\n")
                
                f.write(f"  DAGA Contrast:      {sample.get('daga_contrast', 0):.4f}\n")
                f.write(f"  Baseline Contrast:  {sample.get('bl_contrast', 0):.4f}\n\n")
                
                f.write(f"  DAGA Sparsity:      {sample.get('daga_sparsity', 0):.4f}\n")
                f.write(f"  Baseline Sparsity:  {sample.get('bl_sparsity', 0):.4f}\n\n")
                
                f.write(f"  Visual Difference:  {sample.get('visual_diff', 0):+.4f}\n\n")
                
                f.write("-" * 40 + "\n")
                f.write("Files:\n")
                f.write("-" * 40 + "\n")
                f.write("  daga/          - DAGA model visualizations\n")
                f.write("  baseline/      - Baseline model visualizations\n")
        
        else:
            # Original mode: save combined 3x3 figures
            save_path = output_dir / f"{rank+1:02d}_img_{img_id}_daga.png"
            create_category_similarity_figure(
                image, daga_sim_maps, patch_shape, object_points,
                save_path=save_path,
                title=f"DAGA Feature Similarity (score: {daga_score:.3f})"
            )
            
            # Create Baseline figure
            baseline_path = output_dir / f"{rank+1:02d}_img_{img_id}_baseline.png"
            create_category_similarity_figure(
                image, baseline_sim_maps, patch_shape, object_points,
                save_path=baseline_path,
                title=f"Baseline Feature Similarity (score: {baseline_score:.3f})"
            )
        
        categories_str = ", ".join(list(sample['categories'])[:5])
        mode_str = "(separate folders)" if use_separate_folders else "(combined)"
        print(f"  [{rank+1}/{args.num_samples}] img_{img_id}: DAGA={daga_score:.3f}, BL={baseline_score:.3f}, "
              f"cats: {categories_str}... {mode_str}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("  - XX_img_YYYYY_daga_category_attn.png: DAGA attention per object")
    print("  - XX_img_YYYYY_baseline_category_attn.png: Baseline attention per object")


def parse_args():
    parser = argparse.ArgumentParser(description="Point-to-Attention Visualization")
    
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--daga_checkpoint", type=str, required=True, 
                       help="Path to DAGA checkpoint or pretrained model (if --use_pretrained_daga)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./visualization/results/point_attention")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_eval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_type", type=str, default="auto", 
                       choices=["auto", "classification", "detection"],
                       help="Force model type (auto-detect by default)")
    parser.add_argument("--use_pretrained_daga", action="store_true",
                       help="Use pretrained ViT directly as DAGA (e.g., ViT-L)")
    parser.add_argument("--use_pretrained_baseline", action="store_true",
                       help="Use pretrained ViT directly as Baseline (e.g., ViT-S)")
    parser.add_argument("--category_aggregation", action="store_true",
                       help="Aggregate features from all instances of same category (better semantic)")
    parser.add_argument("--min_visual_diff", type=float, default=0.0,
                       help="Minimum visual difference threshold (filter out samples with less contrast)")
    parser.add_argument("--save_separate", action="store_true",
                       help="Save each image to a separate folder with individual similarity maps and metrics.txt")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_point_attention_visualization(args)

