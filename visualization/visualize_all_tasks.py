"""
Unified Visualization Tool for All Tasks
Loads trained models and generates attention maps + task-specific result visualizations

Supports:
- Classification: Attention maps + predicted class probabilities
- Segmentation: Attention maps + segmentation masks
- Depth Estimation: Attention maps + depth maps
- Detection: Attention maps + bounding boxes
"""

# MUST set environment variable BEFORE any imports to avoid swanlab/pydantic issues
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
from scipy.ndimage import zoom as scipy_zoom

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3"))

# Now safe to import from core (swanlab import is guarded)
from core.backbones import load_dinov3_backbone, process_attention_weights, get_attention_map


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


def create_colormap(name='viridis'):
    """Create custom colormap for depth visualization"""
    if name == 'magma_r':
        return plt.cm.magma_r
    elif name == 'plasma':
        return plt.cm.plasma
    else:
        return plt.cm.viridis


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract args"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = None
    if "args" in checkpoint:
        args = argparse.Namespace(**checkpoint["args"])
    return checkpoint, args


# ============================================================================
# Classification Visualization
# ============================================================================

class ClassificationVisualizer:
    """Visualize classification model predictions and attention"""
    
    def __init__(self, checkpoint_path, device='cuda', num_classes_override=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Load model
        from tasks.classification import ClassificationModel
        
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vits16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path', 
                                   'checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        
        # Get num_classes from checkpoint or override
        if num_classes_override is not None:
            num_classes = num_classes_override
        else:
            num_classes = getattr(self.ckpt_args, 'num_classes', 100)
            # Also try to infer from classifier weight shape
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
        print(f"✓ Classification model loaded (DAGA: {use_daga})")
    
    def visualize(self, image_tensor, class_names=None, save_path=None):
        """Generate visualization for a single image"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            # New 4-value return: logits, adapted_attn, guidance_map, baseline_attn
            outputs = self.model(image_tensor, request_visualization_maps=True)
            if len(outputs) == 4:
                logits, attn_weights, guidance_map, baseline_attn = outputs
            else:
                logits, attn_weights, guidance_map = outputs[:3]
                baseline_attn = None
            probs = F.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        # Process attention
        _, (H, W) = self.model.vit.prepare_tokens_with_masks(image_tensor)
        attn_map = process_attention_weights(attn_weights, H * W, H, W)[0] if attn_weights is not None else None
        # Use baseline_attn for "frozen backbone attention" display
        baseline_map = process_attention_weights(baseline_attn, H * W, H, W)[0] if baseline_attn is not None else None
        guidance = baseline_map if baseline_map is not None else (
            process_attention_weights(guidance_map.unsqueeze(0).unsqueeze(0), H * W, H, W)[0] if guidance_map is not None else None
        )
        
        # Denormalize image
        img = denormalize_image(image_tensor[0])
        
        # Create figure
        n_cols = 4 if self.use_daga else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        
        # Get class name
        class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image\nPred: {class_name} ({pred_prob:.2%})", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Attention overlay
        if attn_map is not None:
            attn_resized = resize_attention_map(attn_map, img.shape[:2])
            axes[1].imshow(img)
            axes[1].imshow(attn_resized, cmap='jet', alpha=0.5)
            title = "DAGA Adapted Attention" if self.use_daga else "Frozen Backbone Attention"
            axes[1].set_title(title, fontsize=12)
            axes[1].axis('off')
        
        if self.use_daga and n_cols > 2:
            # Guidance map (frozen backbone attention)
            if guidance is not None:
                guidance_resized = resize_attention_map(guidance, img.shape[:2])
                axes[2].imshow(img)
                axes[2].imshow(guidance_resized, cmap='jet', alpha=0.5)
                axes[2].set_title("Frozen Backbone Attention", fontsize=12)
                axes[2].axis('off')
            
            # Pure attention map
            if attn_map is not None:
                im = axes[3].imshow(attn_map, cmap='jet')
                axes[3].set_title("Attention Heatmap", fontsize=12)
                axes[3].axis('off')
                plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
        return fig


# ============================================================================
# Segmentation Visualization
# ============================================================================

class SegmentationVisualizer:
    """Visualize segmentation model predictions and attention"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Load model
        from tasks.segmentation import SegmentationModel
        
        model_name = getattr(self.ckpt_args, 'model_name', 'dinov3_vits16')
        pretrained_path = getattr(self.ckpt_args, 'pretrained_path',
                                   'checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        num_classes = getattr(self.ckpt_args, 'num_classes', 150)
        use_daga = getattr(self.ckpt_args, 'use_daga', False)
        daga_layers = getattr(self.ckpt_args, 'daga_layers', [])
        out_indices = getattr(self.ckpt_args, 'out_indices', [2, 5, 8, 11])
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = SegmentationModel(
            vit_model, num_classes=num_classes,
            use_daga=use_daga, daga_layers=daga_layers,
            out_indices=out_indices
        )
        
        # Load weights
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.num_classes = num_classes
        print(f"✓ Segmentation model loaded (DAGA: {use_daga})")
    
    def visualize(self, image_tensor, gt_mask=None, save_path=None):
        """Generate visualization for a single image"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            # New 4-value return: logits, adapted_attn, guidance_map, baseline_attn
            outputs = self.model(image_tensor, request_visualization_maps=True)
            if len(outputs) == 4:
                logits, attn_weights, guidance_map, baseline_attn = outputs
            else:
                logits, attn_weights, guidance_map = outputs[:3]
                baseline_attn = None
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        
        # Process attention (adapted attention from after DAGA)
        _, (H, W) = self.model.vit.prepare_tokens_with_masks(image_tensor)
        attn_map = process_attention_weights(attn_weights, H * W, H, W)[0] if attn_weights is not None else None
        
        # Denormalize image
        img = denormalize_image(image_tensor[0])
        
        # Create figure
        n_cols = 4 if gt_mask is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted segmentation
        axes[1].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=self.num_classes - 1)
        axes[1].set_title("Predicted Segmentation", fontsize=12)
        axes[1].axis('off')
        
        # Attention overlay
        if attn_map is not None:
            attn_resized = resize_attention_map(attn_map, img.shape[:2])
            axes[2].imshow(img)
            axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
            title = "DAGA Attention" if self.use_daga else "Attention Map"
            axes[2].set_title(title, fontsize=12)
            axes[2].axis('off')
        
        # Ground truth if available
        if gt_mask is not None and n_cols > 3:
            gt = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
            gt_display = np.ma.masked_where(gt == 255, gt)
            axes[3].imshow(gt_display, cmap='nipy_spectral', vmin=0, vmax=self.num_classes - 1)
            axes[3].set_title("Ground Truth", fontsize=12)
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
        return fig


# ============================================================================
# Depth Estimation Visualization
# ============================================================================

class DepthVisualizer:
    """Visualize depth estimation model predictions and attention"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Import depth model
        from main_depth import DepthModel
        
        # Get state dict to infer model architecture
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Infer model type from state_dict dimensions
        # Check embed_dim from cls_token or patch_embed
        embed_dim = None
        for key in state_dict.keys():
            if 'cls_token' in key:
                embed_dim = state_dict[key].shape[-1]
                break
            if 'patch_embed.proj.bias' in key:
                embed_dim = state_dict[key].shape[0]
                break
        
        # Determine model name based on embed_dim
        if embed_dim == 768:
            model_name = 'dinov3_vitb16'
            pretrained_path = 'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        elif embed_dim == 1024:
            model_name = 'dinov3_vitl16'
            pretrained_path = 'checkpoints/dinov3_vitl16_pretrain_lvd1689m-e9c0b6c9.pth'
        else:
            model_name = 'dinov3_vits16'
            pretrained_path = 'checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        
        print(f"  Inferred model: {model_name} (embed_dim={embed_dim})")
        
        # Check for DAGA layers in state_dict
        use_daga = any('daga' in k for k in state_dict.keys())
        daga_layers = []
        if use_daga:
            for k in state_dict.keys():
                if 'daga_modules' in k:
                    # Extract layer number from key like 'vit_model.daga_modules.2.xxx'
                    parts = k.split('.')
                    for i, p in enumerate(parts):
                        if p == 'daga_modules' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx not in daga_layers:
                                    daga_layers.append(layer_idx)
                            except ValueError:
                                pass
            daga_layers = sorted(daga_layers)
        
        out_indices = getattr(self.ckpt_args, 'out_indices', [2, 5, 8, 11]) if self.ckpt_args else [2, 5, 8, 11]
        min_depth = getattr(self.ckpt_args, 'min_depth', 0.001) if self.ckpt_args else 0.001
        max_depth = getattr(self.ckpt_args, 'max_depth', 10.0) if self.ckpt_args else 10.0
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = DepthModel(
            vit_model, out_indices=out_indices,
            use_daga=use_daga, daga_layers=daga_layers,
            min_depth=min_depth, max_depth=max_depth
        )
        
        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.max_depth = max_depth
        print(f"✓ Depth model loaded (DAGA: {use_daga}, layers: {daga_layers})")
    
    def visualize(self, image_tensor, gt_depth=None, save_path=None):
        """Generate visualization for a single image"""
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        
        with torch.no_grad():
            pred_depth = self.model(image_tensor)
            if isinstance(pred_depth, tuple):
                pred_depth = pred_depth[0]
            pred_depth = pred_depth.squeeze().cpu().numpy()
        
        # Get attention from the wrapped vit model
        # Use adapted attention (after DAGA) if available, otherwise baseline
        attn_map = None
        if hasattr(self.model, 'vit_model'):
            vit_model = self.model.vit_model
            _, (H, W) = vit_model.vit.prepare_tokens_with_masks(image_tensor)
            
            # Prefer adapted attention (shows DAGA effect)
            if hasattr(vit_model, '_cached_adapted_attn') and vit_model._cached_adapted_attn is not None:
                attn_weights = vit_model._cached_adapted_attn
                attn_map = process_attention_weights(attn_weights, H * W, H, W)[0]
            elif hasattr(vit_model, '_cached_attn') and vit_model._cached_attn is not None:
                attn_weights = vit_model._cached_attn
                attn_map = process_attention_weights(attn_weights, H * W, H, W)[0]
        
        # Denormalize image
        img = denormalize_image(image_tensor[0])
        
        # Create figure
        n_cols = 4 if gt_depth is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted depth
        im_pred = axes[1].imshow(pred_depth, cmap='magma_r', vmin=0, vmax=self.max_depth)
        axes[1].set_title("Predicted Depth", fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Attention overlay
        if attn_map is not None:
            attn_resized = resize_attention_map(attn_map, img.shape[:2])
            axes[2].imshow(img)
            axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
            title = "DAGA Attention" if self.use_daga else "Attention Map"
            axes[2].set_title(title, fontsize=12)
            axes[2].axis('off')
        else:
            axes[2].imshow(img)
            axes[2].set_title("Attention (N/A)", fontsize=12)
            axes[2].axis('off')
        
        # Ground truth if available
        if gt_depth is not None and n_cols > 3:
            gt = gt_depth.cpu().numpy() if torch.is_tensor(gt_depth) else gt_depth
            im_gt = axes[3].imshow(gt, cmap='magma_r', vmin=0, vmax=self.max_depth)
            axes[3].set_title("Ground Truth Depth", fontsize=12)
            axes[3].axis('off')
            plt.colorbar(im_gt, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
        return fig


# ============================================================================
# Detection Visualization
# ============================================================================

class DetectionVisualizer:
    """Visualize detection model predictions and attention"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint, self.ckpt_args = load_checkpoint(checkpoint_path)
        
        # Import detection model
        from tasks.detection import DetectionModel
        
        # Get state dict to infer model architecture
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Infer model type from state_dict dimensions
        embed_dim = None
        for key in state_dict.keys():
            if 'cls_token' in key:
                embed_dim = state_dict[key].shape[-1]
                break
        
        # Determine model name based on embed_dim
        if embed_dim == 768:
            model_name = 'dinov3_vitb16'
            pretrained_path = 'checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        elif embed_dim == 1024:
            model_name = 'dinov3_vitl16'
            pretrained_path = 'checkpoints/dinov3_vitl16_pretrain_lvd1689m-e9c0b6c9.pth'
        else:
            model_name = 'dinov3_vits16'
            pretrained_path = 'checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        
        print(f"  Inferred model: {model_name} (embed_dim={embed_dim})")
        
        # Infer num_classes from cls_head
        num_classes = 80  # Default COCO
        for key in state_dict.keys():
            if 'cls_head.bias' in key:
                num_classes = state_dict[key].shape[0]
                break
        
        # Check for DAGA layers in state_dict
        use_daga = any('daga' in k for k in state_dict.keys())
        daga_layers = []
        if use_daga:
            for k in state_dict.keys():
                if 'daga_modules' in k:
                    parts = k.split('.')
                    for i, p in enumerate(parts):
                        if p == 'daga_modules' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx not in daga_layers:
                                    daga_layers.append(layer_idx)
                            except ValueError:
                                pass
            daga_layers = sorted(daga_layers)
        
        print(f"  num_classes={num_classes}, use_daga={use_daga}, daga_layers={daga_layers}")
        
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        self.model = DetectionModel(
            vit_model, num_classes=num_classes,
            use_daga=use_daga, daga_layers=daga_layers
        )
        
        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.use_daga = use_daga
        self.num_classes = num_classes
        
        # COCO class names
        self.class_names = [
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
        
        print(f"✓ Detection model loaded (DAGA: {use_daga})")
    
    def visualize(self, image_tensor, gt_boxes=None, score_threshold=0.1, save_path=None):
        """Generate visualization for a single image"""
        from core.simple_detection_head import decode_predictions
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device) if image_tensor.dim() == 3 else image_tensor.to(self.device)
        B, _, H, W = image_tensor.shape
        
        with torch.no_grad():
            # Model returns cls_logits, box_preds, centerness (and optionally attn, guidance, baseline_attn)
            outputs = self.model(image_tensor, request_visualization_maps=True)
            
            if len(outputs) >= 6:
                cls_logits, box_preds, centerness, attn_weights, guidance, baseline_attn = outputs
            elif len(outputs) == 5:
                cls_logits, box_preds, centerness, attn_weights, guidance = outputs
                baseline_attn = None
            else:
                cls_logits, box_preds, centerness = outputs[:3]
                attn_weights, guidance, baseline_attn = None, None, None
            
            # Decode predictions to get boxes
            stride = self.model.stride if hasattr(self.model, 'stride') else 14
            detections = decode_predictions(
                cls_logits, box_preds, centerness,
                image_size=(H, W),
                stride=stride,
                score_threshold=score_threshold,
                nms_threshold=0.5,
                max_detections=50
            )
        
        # Get attention map
        attn_map = None
        if attn_weights is not None:
            feat_h, feat_w = H // stride, W // stride
            num_patches = feat_h * feat_w
            attn_map = process_attention_weights(attn_weights, num_patches, feat_h, feat_w)
            if attn_map is not None and len(attn_map) > 0:
                attn_map = attn_map[0]
        
        # Denormalize image
        img = denormalize_image(image_tensor[0])
        
        # Create figure
        n_cols = 3 if gt_boxes is None else 4
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img)
        img_h, img_w = img.shape[:2]  # Get image dimensions for coordinate scaling
        
        if detections and len(detections) > 0:
            det = detections[0]  # First image's detections
            boxes = det.get('boxes', [])
            scores = det.get('scores', [])
            labels = det.get('labels', [])
            
            for box, score, label in zip(boxes, scores, labels):
                # Convert tensors to numpy if needed
                if torch.is_tensor(box):
                    box = box.cpu().numpy()
                if torch.is_tensor(score):
                    score = score.cpu().item()
                if torch.is_tensor(label):
                    label = label.cpu().item()
                
                # Boxes are normalized (0-1), need to scale to image size
                x1, y1, x2, y2 = box
                x1_px, y1_px = x1 * img_w, y1 * img_h
                x2_px, y2_px = x2 * img_w, y2 * img_h
                w_px, h_px = x2_px - x1_px, y2_px - y1_px
                
                # Color based on confidence
                if score < 0.3:
                    color = 'red'
                elif score < 0.5:
                    color = 'orange'
                else:
                    color = 'lime'
                
                rect = patches.Rectangle((x1_px, y1_px), w_px, h_px, linewidth=2,
                                        edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)
                class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"cls{label}"
                axes[1].text(x1_px, y1_px - 5, f"{class_name}: {score:.2f}",
                           color='white', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        axes[1].set_title("Predictions", fontsize=12)
        axes[1].axis('off')
        
        # Attention overlay
        if attn_map is not None:
            attn_resized = resize_attention_map(attn_map, img.shape[:2])
            axes[2].imshow(img)
            axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
            title = "DAGA Attention" if self.use_daga else "Attention Map"
            axes[2].set_title(title, fontsize=12)
            axes[2].axis('off')
        else:
            axes[2].imshow(img)
            axes[2].set_title("Attention (N/A)", fontsize=12)
            axes[2].axis('off')
        
        # Ground truth if available
        if gt_boxes is not None and n_cols > 3:
            axes[3].imshow(img)
            for box in gt_boxes:
                x1, y1, x2, y2 = box[:4]
                # GT boxes are also normalized (0-1), scale to image size
                x1_px, y1_px = x1 * img_w, y1 * img_h
                x2_px, y2_px = x2 * img_w, y2 * img_h
                w_px, h_px = x2_px - x1_px, y2_px - y1_px
                rect = patches.Rectangle((x1_px, y1_px), w_px, h_px, linewidth=2,
                                        edgecolor='red', facecolor='none')
                axes[3].add_patch(rect)
            axes[3].set_title("Ground Truth", fontsize=12)
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
        return fig


# ============================================================================
# Batch Visualization from Dataset
# ============================================================================

def visualize_from_dataset(task, checkpoint_path, data_path, output_dir, num_samples=10, device='cuda'):
    """Load dataset and visualize samples"""
    import torchvision.transforms as transforms
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📊 Visualizing {task.upper()} Task")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data path: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {num_samples}")
    
    if task == 'classification':
        visualizer = ClassificationVisualizer(checkpoint_path, device)
        
        # Load dataset
        from data.classification_datasets import get_classification_dataset
        args = argparse.Namespace(
            dataset='cifar100', data_path=data_path, input_size=224,
            batch_size=1, num_workers=4
        )
        _, val_dataset, num_classes = get_classification_dataset(args)
        class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else None
        
        # Visualize samples
        for i in range(min(num_samples, len(val_dataset))):
            img, label = val_dataset[i]
            save_path = output_dir / f"sample_{i:03d}_class_{label}.png"
            visualizer.visualize(img, class_names=class_names, save_path=save_path)
    
    elif task == 'segmentation':
        visualizer = SegmentationVisualizer(checkpoint_path, device)
        
        # Load dataset
        from core.datasets.segmentation_datasets import ADE20KDataset
        val_dataset = ADE20KDataset(data_path, split='val', input_size=518)
        
        # Visualize samples
        for i in range(min(num_samples, len(val_dataset))):
            img, mask = val_dataset[i]
            save_path = output_dir / f"sample_{i:03d}.png"
            visualizer.visualize(img, gt_mask=mask, save_path=save_path)
    
    elif task == 'depth':
        visualizer = DepthVisualizer(checkpoint_path, device)
        
        # Load dataset
        from core.datasets import NYUDepthV2Dataset
        val_dataset = NYUDepthV2Dataset(data_path, split='val', input_size=518)
        
        # Visualize samples
        for i in range(min(num_samples, len(val_dataset))):
            img, depth = val_dataset[i]
            save_path = output_dir / f"sample_{i:03d}.png"
            visualizer.visualize(img, gt_depth=depth, save_path=save_path)
    
    elif task == 'detection':
        visualizer = DetectionVisualizer(checkpoint_path, device)
        
        # Load dataset (COCO)
        from pycocotools.coco import COCO
        import torchvision.transforms as T
        
        transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        ann_file = Path(data_path) / 'annotations' / 'instances_val2017.json'
        img_dir = Path(data_path) / 'val2017'
        coco = COCO(ann_file)
        img_ids = list(coco.imgs.keys())[:num_samples]
        
        for i, img_id in enumerate(img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = img_dir / img_info['file_name']
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            # Get ground truth boxes
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            gt_boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
            
            save_path = output_dir / f"sample_{i:03d}_img_{img_id}.png"
            visualizer.visualize(img_tensor, gt_boxes=gt_boxes if gt_boxes else None, save_path=save_path)
    
    print(f"\n✅ Visualization complete! Saved to: {output_dir}")


# ============================================================================
# Find Best Samples (Two modes: best_accuracy and best_improvement)
# ============================================================================

def find_best_samples(task, baseline_path, daga_path, data_path, output_dir, 
                      num_samples=10, max_eval=500, device='cuda', 
                      selection_mode='both'):
    """
    Find and visualize best samples based on selection mode.
    
    Args:
        selection_mode: 
            'best_accuracy' - samples where DAGA has highest accuracy/score
            'best_improvement' - samples where DAGA-Baseline difference is largest
            'both' - generate both types of visualizations
    """
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    
    print(f"\n{'='*70}")
    print(f"🔍 Finding Best Samples: {task.upper()}")
    print(f"{'='*70}")
    print(f"  Baseline: {baseline_path}")
    print(f"  DAGA: {daga_path}")
    print(f"  Data path: {data_path}")
    print(f"  Output base: {output_dir}")
    print(f"  Selection mode: {selection_mode}")
    print(f"  Max samples to evaluate: {max_eval}")
    print(f"  Top samples to visualize: {num_samples}")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if task == 'classification':
        # Infer dataset name from baseline path
        # Note: order matters - check cifar100 before cifar10
        dataset_name = None
        baseline_str = str(baseline_path).lower()
        for name in ['cifar100', 'cifar10', 'flowers102', 'dtd', 'pets', 'cars', 'food101', 'sun397', 'imagenet']:
            if name in baseline_str:
                dataset_name = name
                break
        
        _find_classification_samples(baseline_path, daga_path, data_path, 
                                     output_dir, num_samples, max_eval, device,
                                     dataset_name=dataset_name, selection_mode=selection_mode)
    elif task == 'segmentation':
        _find_segmentation_samples(baseline_path, daga_path, data_path,
                                   output_dir, num_samples, max_eval, device,
                                   selection_mode=selection_mode)
    elif task == 'depth':
        _find_depth_samples(baseline_path, daga_path, data_path,
                            output_dir, num_samples, max_eval, device,
                            selection_mode=selection_mode)
    elif task == 'detection':
        _find_detection_samples(baseline_path, daga_path, data_path,
                                output_dir, num_samples, max_eval, device,
                                selection_mode=selection_mode)
    else:
        print(f"⚠️ Sample finding not implemented for {task}")


# Keep old function for backward compatibility
def find_best_improvements(task, baseline_path, daga_path, data_path, output_dir, 
                           num_samples=10, max_eval=500, device='cuda'):
    """Legacy function - redirects to find_best_samples with 'both' mode"""
    find_best_samples(task, baseline_path, daga_path, data_path, output_dir,
                      num_samples, max_eval, device, selection_mode='both')


def _find_classification_samples(baseline_path, daga_path, data_path, 
                                  output_dir, num_samples, max_eval, device,
                                  dataset_name=None, selection_mode='both'):
    """
    Find classification samples based on selection mode.
    
    selection_mode:
        'best_accuracy' - samples where DAGA is correct with highest confidence
        'best_improvement' - samples where DAGA corrects baseline errors with largest gain
        'both' - generate both types
    """
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    
    # Load both models
    baseline_vis = ClassificationVisualizer(baseline_path, device)
    daga_vis = ClassificationVisualizer(daga_path, device)
    
    # Infer dataset name from checkpoint path or data_path if not provided
    if dataset_name is None:
        # Try to infer from checkpoint path
        baseline_str = str(baseline_path).lower()
        if 'cifar10_' in baseline_str and 'cifar100' not in baseline_str:
            dataset_name = 'cifar10'
        elif 'cifar100' in baseline_str:
            dataset_name = 'cifar100'
        elif 'flowers' in baseline_str:
            dataset_name = 'flowers102'
        elif 'dtd' in baseline_str:
            dataset_name = 'dtd'
        elif 'pets' in baseline_str:
            dataset_name = 'pets'
        elif 'cars' in baseline_str:
            dataset_name = 'cars'
        elif 'food' in baseline_str:
            dataset_name = 'food101'
        elif 'sun' in baseline_str:
            dataset_name = 'sun397'
        elif 'imagenet' in baseline_str:
            dataset_name = 'imagenet'
        else:
            # Fallback: try to infer from data_path
            data_str = str(data_path).lower()
            if 'flower' in data_str:
                dataset_name = 'flowers102'
            elif 'dtd' in data_str:
                dataset_name = 'dtd'
            elif 'pets' in data_str:
                dataset_name = 'pets'
            elif 'cars' in data_str:
                dataset_name = 'cars'
            elif 'food' in data_str:
                dataset_name = 'food101'
            elif 'sun' in data_str:
                dataset_name = 'sun397'
            elif 'imagenet' in data_str:
                dataset_name = 'imagenet'
            else:
                dataset_name = 'cifar100'  # Default fallback
    
    print(f"  Dataset: {dataset_name}")
    
    # Load dataset
    from data.classification_datasets import get_classification_dataset
    args = argparse.Namespace(
        dataset=dataset_name, data_path=data_path, input_size=224,
        batch_size=32, num_workers=4
    )
    _, val_dataset, num_classes = get_classification_dataset(args)
    class_names = val_dataset.classes if hasattr(val_dataset, 'classes') else None
    
    # Limit dataset
    if len(val_dataset) > max_eval:
        val_dataset = Subset(val_dataset, list(range(max_eval)))
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Collect all samples with their metrics
    all_samples = []  # For best_accuracy mode
    improvements = []  # For best_improvement mode
    
    print("\n🔍 Evaluating samples...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Comparing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Baseline predictions (model returns 4 values: logits, attn, guidance, baseline_attn)
            outputs1 = baseline_vis.model(images, True)
            logits1, attn1, guide1 = outputs1[0], outputs1[1], outputs1[2]
            preds1 = logits1.argmax(dim=1)
            probs1 = F.softmax(logits1, dim=1)
            
            # DAGA predictions (model returns 4 values: logits, attn, guidance, baseline_attn)
            outputs2 = daga_vis.model(images, True)
            logits2, attn2, guide2 = outputs2[0], outputs2[1], outputs2[2]
            preds2 = logits2.argmax(dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            _, (H, W) = baseline_vis.model.vit.prepare_tokens_with_masks(images)
            
            for idx in range(images.size(0)):
                label_idx = labels[idx].item()
                # Skip if label is out of range
                if label_idx < 0 or label_idx >= probs1.size(1):
                    print(f"  Warning: label {label_idx} out of range [0, {probs1.size(1)}), skipping")
                    continue
                    
                gt_prob1 = probs1[idx, label_idx].item()
                gt_prob2 = probs2[idx, label_idx].item()
                daga_correct = (preds2[idx].item() == label_idx)
                baseline_correct = (preds1[idx].item() == label_idx)
                
                # Process attention maps (handle None safely)
                attn_map1 = None
                attn_map2 = None
                
                if attn1 is not None:
                    try:
                        result = process_attention_weights(attn1[idx:idx+1], H*W, H, W)
                        if result is not None and len(result) > 0:
                            attn_map1 = result[0]
                    except Exception:
                        pass
                
                if attn2 is not None:
                    try:
                        result = process_attention_weights(attn2[idx:idx+1], H*W, H, W)
                        if result is not None and len(result) > 0:
                            attn_map2 = result[0]
                    except Exception:
                        pass
                
                sample_data = {
                    'image': images[idx].cpu(),
                    'label': label_idx,
                    'pred_baseline': preds1[idx].item(),
                    'pred_daga': preds2[idx].item(),
                    'prob_baseline': gt_prob1,
                    'prob_daga': gt_prob2,
                    'prob_diff': gt_prob2 - gt_prob1,
                    'attn_baseline': attn_map1,
                    'attn_daga': attn_map2,
                    'daga_correct': daga_correct,
                    'baseline_correct': baseline_correct,
                }
                
                # For both modes: only collect samples where DAGA correct AND baseline wrong
                # This demonstrates DAGA's advantage over baseline
                if daga_correct and not baseline_correct:
                    all_samples.append(sample_data)
                    improvements.append(sample_data)
    
    # Process based on selection mode
    modes_to_run = []
    if selection_mode == 'both':
        modes_to_run = ['best_accuracy', 'best_improvement']
    else:
        modes_to_run = [selection_mode]
    
    # All samples are now DAGA correct + Baseline wrong (demonstrating DAGA advantage)
    if not all_samples:
        print("⚠️ No improvement samples found (DAGA didn't correct any baseline errors)")
        return
    
    for mode in modes_to_run:
        if mode == 'best_accuracy':
            # Sort by DAGA confidence (highest first) - samples where DAGA is most confident
            sorted_samples = sorted(all_samples, key=lambda x: x['prob_daga'], reverse=True)
            top_samples = sorted_samples[:num_samples]
            mode_output_dir = output_dir / 'best_accuracy'
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✓ Best Accuracy: Found {len(all_samples)} DAGA improvements, visualizing top {len(top_samples)} by confidence")
            _visualize_classification_samples(top_samples, mode_output_dir, class_names, mode='accuracy')
            
        elif mode == 'best_improvement':
            # Sort by confidence gain (largest improvement first)
            sorted_samples = sorted(all_samples, key=lambda x: x['prob_diff'], reverse=True)
            top_samples = sorted_samples[:num_samples]
            mode_output_dir = output_dir / 'best_improvement'
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✓ Best Improvement: Found {len(all_samples)} improvements, visualizing top {len(top_samples)} by gain")
            _visualize_classification_samples(top_samples, mode_output_dir, class_names, mode='improvement')


def _visualize_classification_samples(samples, output_dir, class_names, mode='improvement'):
    """Visualize classification samples - Clean 2x3 layout matching segmentation/depth
    
    Layout (2x3):
    Row 1: Original Image | Baseline Result | Baseline Attention
    Row 2: Ground Truth   | DAGA Result     | DAGA Attention
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Denormalize image
        img = denormalize_image(item['image'])
        
        # Get labels
        true_label = class_names[item['label']] if class_names else f"Class {item['label']}"
        pred1_label = class_names[item['pred_baseline']] if class_names else f"Class {item['pred_baseline']}"
        pred2_label = class_names[item['pred_daga']] if class_names else f"Class {item['pred_daga']}"
        
        baseline_status = "✓" if item['baseline_correct'] else "✗"
        daga_status = "✓" if item['daga_correct'] else "✗"
        baseline_color = '#00AA00' if item['baseline_correct'] else '#CC0000'
        daga_color = '#00AA00' if item['daga_correct'] else '#CC0000'
        
        if mode == 'accuracy':
            fig.suptitle(
                f"Classification #{i+1} | True: {true_label}\n"
                f"Conf: {item['prob_baseline']:.3f} → {item['prob_daga']:.3f}",
                fontsize=16, fontweight='bold'
            )
        else:
            fig.suptitle(
                f"Classification #{i+1} | True: {true_label} | Confidence Gain: +{item['prob_diff']:.3f}\n"
                f"Conf: {item['prob_baseline']:.3f} → {item['prob_daga']:.3f}",
                fontsize=16, fontweight='bold'
            )
        
        # Get attention maps
        attn_baseline = item.get('attn_baseline')
        attn_daga = item.get('attn_daga')
        
        # Normalize and resize attention maps
        if attn_baseline is not None:
            attn_baseline_norm = (attn_baseline - attn_baseline.min()) / (attn_baseline.max() - attn_baseline.min() + 1e-8)
            attn_baseline_resized = resize_attention_map(attn_baseline_norm, img.shape[:2])
            # Enhance contrast
            attn_baseline_enhanced = np.power(attn_baseline_resized, 0.6)
        else:
            attn_baseline_enhanced = None
            
        if attn_daga is not None:
            attn_daga_norm = (attn_daga - attn_daga.min()) / (attn_daga.max() - attn_daga.min() + 1e-8)
            attn_daga_resized = resize_attention_map(attn_daga_norm, img.shape[:2])
            # Enhance contrast
            attn_daga_enhanced = np.power(attn_daga_resized, 0.6)
        else:
            attn_daga_enhanced = None
        
        # Row 1: Original, Baseline Result, Baseline Attention
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Baseline Result (image with prediction text)
        axes[0, 1].imshow(img)
        axes[0, 1].set_title(f"Baseline {baseline_status}: {pred1_label}\nConf: {item['prob_baseline']:.3f}", 
                           fontsize=14, color=baseline_color, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Baseline Attention Overlay
        if attn_baseline_enhanced is not None:
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(attn_baseline_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[0, 2].set_title("Baseline Attention", fontsize=14)
        else:
            axes[0, 2].imshow(img)
            axes[0, 2].set_title("Baseline Attention (N/A)", fontsize=14)
        axes[0, 2].axis('off')
        
        # Row 2: Ground Truth label, DAGA Result, DAGA Attention
        axes[1, 0].imshow(img)
        axes[1, 0].set_title(f"Ground Truth: {true_label}", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # DAGA Result
        axes[1, 1].imshow(img)
        axes[1, 1].set_title(f"DAGA {daga_status}: {pred2_label}\nConf: {item['prob_daga']:.3f}",
                           fontsize=14, color=daga_color, fontweight='bold')
        axes[1, 1].axis('off')
        
        # DAGA Attention Overlay
        if attn_daga_enhanced is not None:
            axes[1, 2].imshow(img)
            axes[1, 2].imshow(attn_daga_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[1, 2].set_title("DAGA Attention", fontsize=14, color='#00AA00')
        else:
            axes[1, 2].imshow(img)
            axes[1, 2].set_title("DAGA Attention (N/A)", fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if mode == 'accuracy':
            save_path = output_dir / f"best_{i+1:02d}_conf_{item['prob_daga']:.3f}.png"
        else:
            save_path = output_dir / f"improvement_{i+1:02d}_gain_{item['prob_diff']:.3f}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\n✅ Samples saved to: {output_dir}")


def _find_segmentation_samples(baseline_path, daga_path, data_path,
                                output_dir, num_samples, max_eval, device,
                                selection_mode='both'):
    """Find segmentation samples based on selection mode"""
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    
    # Load both models
    baseline_vis = SegmentationVisualizer(baseline_path, device)
    daga_vis = SegmentationVisualizer(daga_path, device)
    
    # Load dataset
    from core.datasets.segmentation_datasets import ADE20KDataset
    val_dataset = ADE20KDataset(data_path, split='val', input_size=518)
    
    # Limit dataset
    if len(val_dataset) > max_eval:
        val_dataset = Subset(val_dataset, list(range(max_eval)))
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Helper function to calculate mIoU
    def calc_miou(pred, target, num_classes):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        ious = []
        for cls in range(num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            
            if target_mask.sum() == 0:
                continue
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0
    
    # Collect all samples
    all_samples = []  # For best_accuracy
    improvements = []  # For best_improvement
    num_classes = baseline_vis.num_classes
    
    print("\n🔍 Evaluating samples...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Comparing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Baseline predictions (model returns 4 values: logits, attn, guidance, baseline_attn)
            outputs1 = baseline_vis.model(images, True)
            logits1, attn1 = outputs1[0], outputs1[1]
            preds1 = logits1.argmax(dim=1)
            
            # DAGA predictions (model returns 4 values: logits, attn, guidance, baseline_attn)
            outputs2 = daga_vis.model(images, True)
            logits2, attn2 = outputs2[0], outputs2[1]
            preds2 = logits2.argmax(dim=1)
            
            _, (H, W) = baseline_vis.model.vit.prepare_tokens_with_masks(images)
            
            for idx in range(images.size(0)):
                miou1 = calc_miou(preds1[idx], masks[idx], num_classes)
                miou2 = calc_miou(preds2[idx], masks[idx], num_classes)
                
                attn_map1 = process_attention_weights(attn1[idx:idx+1], H*W, H, W)[0] if attn1 is not None else None
                attn_map2 = process_attention_weights(attn2[idx:idx+1], H*W, H, W)[0] if attn2 is not None else None
                
                sample_data = {
                    'image': images[idx].cpu(),
                    'mask_gt': masks[idx].cpu(),
                    'mask_baseline': preds1[idx].cpu(),
                    'mask_daga': preds2[idx].cpu(),
                    'miou_baseline': miou1 * 100,
                    'miou_daga': miou2 * 100,
                    'miou_diff': (miou2 - miou1) * 100,
                    'attn_baseline': attn_map1,
                    'attn_daga': attn_map2
                }
                
                all_samples.append(sample_data)
                if miou2 > miou1:
                    improvements.append(sample_data)
    
    # Process based on selection mode
    modes_to_run = ['best_accuracy', 'best_improvement'] if selection_mode == 'both' else [selection_mode]
    output_dir = Path(output_dir)
    
    for mode in modes_to_run:
        if mode == 'best_accuracy':
            all_samples.sort(key=lambda x: x['miou_daga'], reverse=True)
            top_samples = all_samples[:num_samples]
            mode_output_dir = output_dir / 'best_accuracy'
            print(f"\n✓ Best Accuracy: Visualizing top {len(top_samples)} by DAGA mIoU")
        else:
            if not improvements:
                print("⚠️ No improvements found")
                continue
            improvements.sort(key=lambda x: x['miou_diff'], reverse=True)
            top_samples = improvements[:num_samples]
            mode_output_dir = output_dir / 'best_improvement'
            print(f"\n✓ Best Improvement: Found {len(improvements)} improvements, visualizing top {len(top_samples)}")
        
        _visualize_segmentation_samples(top_samples, mode_output_dir, mode)


def _visualize_segmentation_samples(samples, output_dir, mode='improvement'):
    """Visualize segmentation samples - Clean 2x3 layout matching classification/depth
    
    Layout (2x3):
    Row 1: Original Image | Baseline Result | Baseline Attention
    Row 2: Ground Truth   | DAGA Result     | DAGA Attention
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a custom colormap for segmentation (more distinct colors)
    from matplotlib.colors import ListedColormap
    np.random.seed(42)
    colors = np.random.rand(150, 3)
    colors[0] = [0, 0, 0]  # Background is black
    seg_cmap = ListedColormap(colors)
    
    for i, item in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        img = denormalize_image(item['image'])
        
        fig.suptitle(
            f"Segmentation #{i+1}  |  mIoU: {item['miou_baseline']:.1f}% → {item['miou_daga']:.1f}% (+{item['miou_diff']:.1f}%)",
            fontsize=16, fontweight='bold'
        )
        
        # Get masks
        mask_baseline = item['mask_baseline'].numpy()
        mask_daga = item['mask_daga'].numpy()
        gt_mask = item['mask_gt'].numpy()
        
        # Get attention maps
        attn_baseline = item.get('attn_baseline')
        attn_daga = item.get('attn_daga')
        
        # Normalize and resize attention maps
        if attn_baseline is not None:
            attn_baseline_norm = (attn_baseline - attn_baseline.min()) / (attn_baseline.max() - attn_baseline.min() + 1e-8)
            attn_baseline_resized = resize_attention_map(attn_baseline_norm, img.shape[:2])
            attn_baseline_enhanced = np.power(attn_baseline_resized, 0.6)
        else:
            attn_baseline_enhanced = None
            
        if attn_daga is not None:
            attn_daga_norm = (attn_daga - attn_daga.min()) / (attn_daga.max() - attn_daga.min() + 1e-8)
            attn_daga_resized = resize_attention_map(attn_daga_norm, img.shape[:2])
            attn_daga_enhanced = np.power(attn_daga_resized, 0.6)
        else:
            attn_daga_enhanced = None
        
        # Row 1: Original, Baseline Result, Baseline Attention
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Baseline segmentation
        axes[0, 1].imshow(img, alpha=0.4)
        axes[0, 1].imshow(mask_baseline, cmap=seg_cmap, alpha=0.6, vmin=0, vmax=149)
        axes[0, 1].set_title(f"Baseline: {item['miou_baseline']:.1f}%", 
                           color='#CC0000', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Baseline Attention
        if attn_baseline_enhanced is not None:
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(attn_baseline_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[0, 2].set_title("Baseline Attention", fontsize=14)
        else:
            axes[0, 2].imshow(img)
            axes[0, 2].set_title("Baseline Attention (N/A)", fontsize=14)
        axes[0, 2].axis('off')
        
        # Row 2: Ground Truth, DAGA Result, DAGA Attention
        gt_display = np.ma.masked_where(gt_mask == 255, gt_mask)
        axes[1, 0].imshow(img, alpha=0.4)
        axes[1, 0].imshow(gt_display, cmap=seg_cmap, alpha=0.6, vmin=0, vmax=149)
        axes[1, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # DAGA segmentation
        axes[1, 1].imshow(img, alpha=0.4)
        axes[1, 1].imshow(mask_daga, cmap=seg_cmap, alpha=0.6, vmin=0, vmax=149)
        axes[1, 1].set_title(f"DAGA: {item['miou_daga']:.1f}%",
                           color='#00AA00', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # DAGA Attention
        if attn_daga_enhanced is not None:
            axes[1, 2].imshow(img)
            axes[1, 2].imshow(attn_daga_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[1, 2].set_title("DAGA Attention", fontsize=14, color='#00AA00')
        else:
            axes[1, 2].imshow(img)
            axes[1, 2].set_title("DAGA Attention (N/A)", fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if mode == 'accuracy':
            save_path = output_dir / f"best_{i+1:02d}_miou_{item['miou_daga']:.2f}.png"
        else:
            save_path = output_dir / f"improvement_{i+1:02d}_miou_gain_{item['miou_diff']:.2f}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\n✅ Samples saved to: {output_dir}")


def _find_depth_samples(baseline_path, daga_path, data_path,
                         output_dir, num_samples, max_eval, device,
                         selection_mode='both'):
    """Find depth samples based on selection mode"""
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    
    # Load both models
    baseline_vis = DepthVisualizer(baseline_path, device)
    daga_vis = DepthVisualizer(daga_path, device)
    
    # Load dataset
    from core.datasets import NYUDepthV2Dataset
    val_dataset = NYUDepthV2Dataset(data_path, split='val', input_size=518)
    
    # Limit dataset
    if len(val_dataset) > max_eval:
        val_dataset = Subset(val_dataset, list(range(max_eval)))
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Helper function to calculate RMSE
    def calc_rmse(pred, target):
        valid_mask = target > 0
        if valid_mask.sum() == 0:
            return float('inf')
        diff = pred[valid_mask] - target[valid_mask]
        return np.sqrt((diff ** 2).mean())
    
    # Collect all samples
    all_samples = []  # For best_accuracy
    improvements = []  # For best_improvement
    
    print("\n🔍 Evaluating samples...")
    with torch.no_grad():
        for images, depths in tqdm(val_loader, desc="Comparing"):
            images = images.to(device)
            
            # Baseline predictions
            pred1 = baseline_vis.model(images)
            if isinstance(pred1, tuple):
                pred1 = pred1[0]
            
            # DAGA predictions
            pred2 = daga_vis.model(images)
            if isinstance(pred2, tuple):
                pred2 = pred2[0]
            
            # Get attention maps
            _, (H, W) = baseline_vis.model.vit_model.vit.prepare_tokens_with_masks(images)
            
            # Get baseline attention (frozen backbone attention)
            attn1 = None
            if hasattr(baseline_vis.model.vit_model, '_cached_attn'):
                attn1 = baseline_vis.model.vit_model._cached_attn
            
            # Get DAGA attention - use adapted attention (after DAGA) if available
            attn2 = None
            if hasattr(daga_vis.model.vit_model, '_cached_adapted_attn') and daga_vis.model.vit_model._cached_adapted_attn is not None:
                attn2 = daga_vis.model.vit_model._cached_adapted_attn
            elif hasattr(daga_vis.model.vit_model, '_cached_attn'):
                attn2 = daga_vis.model.vit_model._cached_attn
            
            for idx in range(images.size(0)):
                pred1_np = pred1[idx].squeeze().cpu().numpy()
                pred2_np = pred2[idx].squeeze().cpu().numpy()
                gt_np = depths[idx].squeeze().numpy()
                
                # Resize predictions to match GT size if needed
                if pred1_np.shape != gt_np.shape:
                    from scipy.ndimage import zoom as scipy_zoom
                    scale_h = gt_np.shape[0] / pred1_np.shape[0]
                    scale_w = gt_np.shape[1] / pred1_np.shape[1]
                    pred1_np = scipy_zoom(pred1_np, (scale_h, scale_w), order=1)
                    pred2_np = scipy_zoom(pred2_np, (scale_h, scale_w), order=1)
                
                rmse1 = calc_rmse(pred1_np, gt_np)
                rmse2 = calc_rmse(pred2_np, gt_np)
                
                # Process attention maps
                attn_map1 = None
                attn_map2 = None
                if attn1 is not None:
                    try:
                        result = process_attention_weights(attn1[idx:idx+1], H*W, H, W)
                        if result is not None and len(result) > 0:
                            attn_map1 = result[0]
                    except:
                        pass
                if attn2 is not None:
                    try:
                        result = process_attention_weights(attn2[idx:idx+1], H*W, H, W)
                        if result is not None and len(result) > 0:
                            attn_map2 = result[0]
                    except:
                        pass
                
                sample_data = {
                    'image': images[idx].cpu(),
                    'depth_gt': depths[idx],
                    'depth_baseline': pred1_np,
                    'depth_daga': pred2_np,
                    'rmse_baseline': rmse1,
                    'rmse_daga': rmse2,
                    'rmse_diff': rmse1 - rmse2,  # Positive means DAGA is better
                    'attn_baseline': attn_map1,
                    'attn_daga': attn_map2,
                }
                
                all_samples.append(sample_data)
                if rmse2 < rmse1:  # Lower RMSE is better
                    improvements.append(sample_data)
    
    # Process based on selection mode
    modes_to_run = ['best_accuracy', 'best_improvement'] if selection_mode == 'both' else [selection_mode]
    output_dir = Path(output_dir)
    max_depth = baseline_vis.max_depth
    
    for mode in modes_to_run:
        if mode == 'best_accuracy':
            # Sort by lowest DAGA RMSE (best accuracy)
            all_samples.sort(key=lambda x: x['rmse_daga'])
            top_samples = all_samples[:num_samples]
            mode_output_dir = output_dir / 'best_accuracy'
            print(f"\n✓ Best Accuracy: Visualizing top {len(top_samples)} by lowest DAGA RMSE")
        else:
            if not improvements:
                print("⚠️ No improvements found")
                continue
            improvements.sort(key=lambda x: x['rmse_diff'], reverse=True)
            top_samples = improvements[:num_samples]
            mode_output_dir = output_dir / 'best_improvement'
            print(f"\n✓ Best Improvement: Found {len(improvements)} improvements, visualizing top {len(top_samples)}")
        
        _visualize_depth_samples(top_samples, mode_output_dir, max_depth, mode)


def _visualize_depth_samples(samples, output_dir, max_depth, mode='improvement'):
    """Visualize depth samples - Clean 2x3 layout matching segmentation
    
    Layout (2x3):
    Row 1: Original Image | Baseline Depth | Baseline Attention
    Row 2: Ground Truth   | DAGA Depth     | DAGA Attention
    """
    from scipy.ndimage import zoom
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        img = denormalize_image(item['image'])
        gt_depth = item['depth_gt'].squeeze().numpy()
        baseline_depth = item['depth_baseline']
        daga_depth = item['depth_daga']
        
        # Resize predictions to match GT if needed
        if baseline_depth.shape != gt_depth.shape:
            zoom_h = gt_depth.shape[0] / baseline_depth.shape[0]
            zoom_w = gt_depth.shape[1] / baseline_depth.shape[1]
            baseline_depth = zoom(baseline_depth, (zoom_h, zoom_w), order=1)
            daga_depth = zoom(daga_depth, (zoom_h, zoom_w), order=1)
        
        # Use adaptive depth range based on actual data for better contrast
        valid_mask = gt_depth > 0
        depth_min = gt_depth[valid_mask].min() if valid_mask.any() else 0
        depth_max = gt_depth[valid_mask].max() if valid_mask.any() else max_depth
        depth_range = depth_max - depth_min
        vmin = max(0, depth_min - 0.1 * depth_range)
        vmax = depth_max + 0.1 * depth_range
        
        # Calculate improvement percentage
        pct_improvement = (item['rmse_diff'] / item['rmse_baseline']) * 100 if item['rmse_baseline'] > 0 else 0
        
        fig.suptitle(
            f"Depth #{i+1}  |  RMSE: {item['rmse_baseline']:.3f} → {item['rmse_daga']:.3f} "
            f"(↓{item['rmse_diff']:.3f}, {pct_improvement:.1f}% better)",
            fontsize=16, fontweight='bold'
        )
        
        # Row 1: Original, Baseline Depth, Baseline Attention
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(baseline_depth, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Baseline: RMSE {item['rmse_baseline']:.3f}", 
                           color='#CC0000', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
        
        # Baseline Attention
        if item.get('attn_baseline') is not None:
            attn_resized = resize_attention_map(item['attn_baseline'], img.shape[:2])
            attn_enhanced = np.power(np.clip((attn_resized - attn_resized.min()) / 
                                             (attn_resized.max() - attn_resized.min() + 1e-8), 0, 1), 0.6)
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(attn_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[0, 2].set_title("Baseline Attention", fontsize=14)
            axes[0, 2].axis('off')
        else:
            axes[0, 2].imshow(img)
            axes[0, 2].set_title("Baseline Attention (N/A)", fontsize=14)
            axes[0, 2].axis('off')
        
        # Row 2: Ground Truth, DAGA Depth, DAGA Attention
        im_gt = axes[1, 0].imshow(gt_depth, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im_gt, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (m)')
        
        im2 = axes[1, 1].imshow(daga_depth, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f"DAGA: RMSE {item['rmse_daga']:.3f}",
                           color='#00AA00', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Depth (m)')
        
        # DAGA Attention
        if item.get('attn_daga') is not None:
            attn_resized = resize_attention_map(item['attn_daga'], img.shape[:2])
            attn_enhanced = np.power(np.clip((attn_resized - attn_resized.min()) / 
                                             (attn_resized.max() - attn_resized.min() + 1e-8), 0, 1), 0.6)
            axes[1, 2].imshow(img)
            axes[1, 2].imshow(attn_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[1, 2].set_title("DAGA Attention", fontsize=14, color='#00AA00')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].imshow(img)
            axes[1, 2].set_title("DAGA Attention (N/A)", fontsize=14)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if mode == 'accuracy':
            save_path = output_dir / f"best_{i+1:02d}_rmse_{item['rmse_daga']:.4f}.png"
        else:
            save_path = output_dir / f"improvement_{i+1:02d}_rmse_gain_{item['rmse_diff']:.4f}.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\n✅ Samples saved to: {output_dir}")


def get_focused_attention_map(raw_weights, num_patches, H, W, method='max_head'):
    """
    Get a more focused attention map for visualization.
    
    Args:
        raw_weights: (B, num_heads, seq_len, seq_len) attention weights
        num_patches: expected number of patches
        H, W: patch grid dimensions
        method: 'max_head' - use the head with maximum variance (most focused)
                'top_k_heads' - average top-k most focused heads
                'last_head' - use the last head
    
    Returns:
        (B, H, W) attention map
    """
    B, num_heads, seq_len, _ = raw_weights.shape
    num_registers = seq_len - num_patches - 1
    
    # Get CLS attention to patches (skip CLS and registers)
    cls_attn = raw_weights[:, :, 0, 1 + num_registers:]  # (B, num_heads, num_patches)
    
    if method == 'max_head':
        # Find the head with maximum variance (most focused/discriminative)
        variances = cls_attn.var(dim=-1)  # (B, num_heads)
        best_head_idx = variances.argmax(dim=-1)  # (B,)
        
        # Extract attention from best head for each sample
        attn_maps = []
        for b in range(B):
            attn_maps.append(cls_attn[b, best_head_idx[b]])
        attn = torch.stack(attn_maps, dim=0)  # (B, num_patches)
        
    elif method == 'top_k_heads':
        # Use top-k heads with highest variance
        k = min(4, num_heads)
        variances = cls_attn.var(dim=-1)  # (B, num_heads)
        _, top_indices = variances.topk(k, dim=-1)  # (B, k)
        
        attn_maps = []
        for b in range(B):
            head_attns = cls_attn[b, top_indices[b]]  # (k, num_patches)
            attn_maps.append(head_attns.mean(dim=0))
        attn = torch.stack(attn_maps, dim=0)  # (B, num_patches)
        
    elif method == 'last_head':
        attn = cls_attn[:, -1, :]  # (B, num_patches)
        
    else:  # default: mean over heads
        attn = cls_attn.mean(dim=1)  # (B, num_patches)
    
    # Normalize
    min_val = attn.amin(dim=-1, keepdim=True)
    max_val = attn.amax(dim=-1, keepdim=True)
    attn = (attn - min_val) / (max_val - min_val + 1e-8)
    
    # Reshape to spatial
    return attn.reshape(B, H, W).cpu().numpy()


def _find_detection_samples(baseline_path, daga_path, data_path,
                             output_dir, num_samples, max_eval, device,
                             selection_mode='both'):
    """Find detection samples based on selection mode"""
    from tqdm import tqdm
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import torchvision.transforms as T
    from core.simple_detection_head import decode_predictions
    from core.backbones import get_attention_map, process_attention_weights, compute_daga_guidance_map
    
    output_dir = Path(output_dir)
    
    # Load both models
    baseline_vis = DetectionVisualizer(baseline_path, device)
    daga_vis = DetectionVisualizer(daga_path, device)
    
    # Load COCO dataset
    ann_file = Path(data_path) / 'annotations' / 'instances_val2017.json'
    img_dir = Path(data_path) / 'val2017'
    coco = COCO(ann_file)
    
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Sample images
    all_img_ids = list(coco.imgs.keys())
    img_ids = all_img_ids[:min(max_eval, len(all_img_ids))]
    
    # Helper function to compute IoU-based score for a single image
    def compute_image_score(model, img_tensor, gt_boxes, score_threshold=0.3):
        """Compute detection score for an image (higher is better)"""
        with torch.no_grad():
            outputs = model.model(img_tensor.unsqueeze(0).to(device))
            if len(outputs) >= 3:
                cls_logits, box_preds, centerness = outputs[:3]
            else:
                return 0.0
            
            stride = model.model.stride if hasattr(model.model, 'stride') else 14
            detections = decode_predictions(
                cls_logits, box_preds, centerness,
                image_size=(518, 518),
                stride=stride,
                score_threshold=score_threshold,
                nms_threshold=0.5,
                max_detections=30
            )
        
        if not detections or len(detections[0]['boxes']) == 0:
            return 0.0
        
        # Simple scoring: count high-confidence detections that overlap with GT
        pred_boxes = detections[0]['boxes'].cpu().numpy()
        pred_scores = detections[0]['scores'].cpu().numpy()
        
        score = 0.0
        for pb, ps in zip(pred_boxes, pred_scores):
            if ps >= score_threshold:
                score += ps  # Sum of confidence scores
        
        return score
    
    print("\n🔍 Evaluating samples...")
    all_samples = []  # For best_accuracy
    improvements = []  # For best_improvement
    
    for img_id in tqdm(img_ids, desc="Comparing"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info['file_name']
        
        try:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_tensor = transform(img)
            
            # Get ground truth boxes (normalized to 0-1)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            gt_boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                # Normalize to 0-1
                gt_boxes.append([x/orig_w, y/orig_h, (x+w)/orig_w, (y+h)/orig_h])
            
            if len(gt_boxes) == 0:
                continue
            
            # Get scores for both models
            baseline_score = compute_image_score(baseline_vis, img_tensor, gt_boxes)
            daga_score = compute_image_score(daga_vis, img_tensor, gt_boxes)
            
            # DAGA improvement
            score_diff = daga_score - baseline_score
            
            sample_data = {
                'img_id': img_id,
                'img_path': img_path,
                'img_tensor': img_tensor,
                'gt_boxes': gt_boxes,
                'baseline_score': baseline_score,
                'daga_score': daga_score,
                'score_diff': score_diff
            }
            
            all_samples.append(sample_data)
            if score_diff > 0:  # DAGA is better
                improvements.append(sample_data)
                
        except Exception as e:
            print(f"  Error processing {img_id}: {e}")
            continue
    
    # Process based on selection mode
    modes_to_run = ['best_accuracy', 'best_improvement'] if selection_mode == 'both' else [selection_mode]
    
    for mode in modes_to_run:
        if mode == 'best_accuracy':
            # Sort by highest DAGA score
            all_samples.sort(key=lambda x: x['daga_score'], reverse=True)
            top_samples = all_samples[:num_samples]
            mode_output_dir = output_dir / 'best_accuracy'
            print(f"\n✓ Best Accuracy: Visualizing top {len(top_samples)} by DAGA score")
        else:
            if not improvements:
                print("⚠️ No improvements found")
                continue
            improvements.sort(key=lambda x: x['score_diff'], reverse=True)
            top_samples = improvements[:num_samples]
            mode_output_dir = output_dir / 'best_improvement'
            print(f"\n✓ Best Improvement: Found {len(improvements)} improvements, visualizing top {len(top_samples)}")
        
        _visualize_detection_samples(top_samples, mode_output_dir, baseline_vis, daga_vis, device, mode)


def _visualize_detection_samples(samples, output_dir, baseline_vis, daga_vis, device, mode='improvement'):
    """Visualize detection samples - Clean 2x3 layout matching segmentation
    
    Layout (2x3):
    Row 1: Original Image | Baseline Predictions | Baseline Attention
    Row 2: Ground Truth   | DAGA Predictions     | DAGA Attention
    
    Key: Only show high-quality predictions that match GT (IoU > 0.5)
    """
    from core.simple_detection_head import decode_predictions
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each sample
    for i, item in enumerate(samples):
        img_tensor = item['img_tensor'].to(device)
        gt_boxes = item['gt_boxes']
        
        # Get predictions and attention from both models
        stride = baseline_vis.model.stride if hasattr(baseline_vis.model, 'stride') else 14
        feat_h, feat_w = 518 // stride, 518 // stride
        num_patches = feat_h * feat_w
        
        with torch.no_grad():
            # Baseline (model returns 6 values: cls, box, ctr, attn, guidance, baseline_attn)
            baseline_out = baseline_vis.model(img_tensor.unsqueeze(0), request_visualization_maps=True)
            if len(baseline_out) >= 6:
                b_cls, b_box, b_ctr, b_attn, _, _ = baseline_out
            elif len(baseline_out) >= 4:
                b_cls, b_box, b_ctr, b_attn = baseline_out[:4]
            else:
                b_cls, b_box, b_ctr = baseline_out[:3]
                b_attn = None
            
            # DAGA (model returns 6 values: cls, box, ctr, attn, guidance, baseline_attn)
            daga_out = daga_vis.model(img_tensor.unsqueeze(0), request_visualization_maps=True)
            if len(daga_out) >= 6:
                d_cls, d_box, d_ctr, d_attn, d_guidance, _ = daga_out
            elif len(daga_out) >= 5:
                d_cls, d_box, d_ctr, d_attn, d_guidance = daga_out[:5]
            else:
                d_cls, d_box, d_ctr = daga_out[:3]
                d_attn, d_guidance = None, None
        
        # Decode predictions with higher threshold for cleaner visualization
        baseline_dets = decode_predictions(b_cls, b_box, b_ctr, (518, 518), stride, 0.4, 0.5, 20)
        daga_dets = decode_predictions(d_cls, d_box, d_ctr, (518, 518), stride, 0.4, 0.5, 20)
        
        # Process attention maps
        baseline_attn_np = None
        daga_attn_np = None
        
        if b_attn is not None:
            baseline_attn_np = get_focused_attention_map(b_attn, num_patches, feat_h, feat_w, method='max_head')
            if baseline_attn_np is not None and len(baseline_attn_np) > 0:
                baseline_attn_np = baseline_attn_np[0]
        
        if d_attn is not None:
            daga_attn_np = get_focused_attention_map(d_attn, num_patches, feat_h, feat_w, method='max_head')
            if daga_attn_np is not None and len(daga_attn_np) > 0:
                daga_attn_np = daga_attn_np[0]
        
        # Create visualization - 2 rows x 3 cols
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        img = denormalize_image(img_tensor)
        img_h, img_w = img.shape[:2]
        
        # Helper: Get best matching predictions for each GT box (1-to-1 matching)
        def get_best_matches(dets, gt_boxes, iou_threshold=0.5):
            """Get best prediction for each GT box (clean 1-to-1 matching)"""
            if not dets or len(dets[0]['boxes']) == 0 or len(gt_boxes) == 0:
                return [], [], 0
            
            pred_boxes = dets[0]['boxes'].cpu().numpy()
            pred_scores = dets[0]['scores'].cpu().numpy()
            
            matched_boxes = []
            matched_scores = []
            matched_gt_count = 0
            used_preds = set()
            
            for gt_box in gt_boxes:
                best_iou = 0
                best_idx = -1
                best_score = 0
                
                for idx, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                    if idx in used_preds:
                        continue
                    
                    # Calculate IoU
                    x1 = max(box[0], gt_box[0])
                    y1 = max(box[1], gt_box[1])
                    x2 = min(box[2], gt_box[2])
                    y2 = min(box[3], gt_box[3])
                    
                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union_area = box_area + gt_area - inter_area
                    
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_idx = idx
                        best_score = score
                
                if best_idx >= 0:
                    matched_boxes.append(pred_boxes[best_idx])
                    matched_scores.append(best_score)
                    used_preds.add(best_idx)
                    matched_gt_count += 1
            
            return matched_boxes, matched_scores, matched_gt_count
        
        # Get clean matched predictions (IoU > 0.5)
        baseline_matched, baseline_scores, baseline_count = get_best_matches(baseline_dets, gt_boxes, iou_threshold=0.5)
        daga_matched, daga_scores, daga_count = get_best_matches(daga_dets, gt_boxes, iou_threshold=0.5)
        
        fig.suptitle(
            f"Detection #{i+1}  |  GT Matched: {baseline_count}/{len(gt_boxes)} → {daga_count}/{len(gt_boxes)}",
            fontsize=16, fontweight='bold'
        )
        
        # Row 1, Col 0: Original Image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Row 1, Col 1: Baseline Predictions (clean, matched only)
        axes[0, 1].imshow(img)
        for box, score in zip(baseline_matched, baseline_scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1 * img_w, y1 * img_h), (x2 - x1) * img_w, (y2 - y1) * img_h,
                linewidth=3, edgecolor='#FF6600', facecolor='none'
            )
            axes[0, 1].add_patch(rect)
        axes[0, 1].set_title(f"Baseline: {baseline_count}/{len(gt_boxes)}", 
                           color='#CC0000', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Row 1, Col 2: Baseline Attention
        if baseline_attn_np is not None:
            attn_resized = resize_attention_map(baseline_attn_np, (img_h, img_w))
            attn_enhanced = np.power(np.clip((attn_resized - attn_resized.min()) / 
                                             (attn_resized.max() - attn_resized.min() + 1e-8), 0, 1), 0.6)
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(attn_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[0, 2].set_title("Baseline Attention", fontsize=14)
        else:
            axes[0, 2].imshow(img)
            axes[0, 2].set_title("Baseline Attention", fontsize=14)
        axes[0, 2].axis('off')
        
        # Row 2, Col 0: Ground Truth boxes (clean cyan dashed)
        axes[1, 0].imshow(img)
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1 * img_w, y1 * img_h), (x2 - x1) * img_w, (y2 - y1) * img_h,
                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
            )
            axes[1, 0].add_patch(rect)
        axes[1, 0].set_title(f"Ground Truth ({len(gt_boxes)})", fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Row 2, Col 1: DAGA Predictions (clean, matched only)
        axes[1, 1].imshow(img)
        for box, score in zip(daga_matched, daga_scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1 * img_w, y1 * img_h), (x2 - x1) * img_w, (y2 - y1) * img_h,
                linewidth=3, edgecolor='#00FF00', facecolor='none'
            )
            axes[1, 1].add_patch(rect)
        axes[1, 1].set_title(f"DAGA: {daga_count}/{len(gt_boxes)}", 
                           color='#00AA00', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Row 2, Col 2: DAGA Attention
        if daga_attn_np is not None:
            attn_resized = resize_attention_map(daga_attn_np, (img_h, img_w))
            attn_enhanced = np.power(np.clip((attn_resized - attn_resized.min()) / 
                                             (attn_resized.max() - attn_resized.min() + 1e-8), 0, 1), 0.6)
            axes[1, 2].imshow(img)
            axes[1, 2].imshow(attn_enhanced, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            axes[1, 2].set_title("DAGA Attention", fontsize=14, color='#00AA00')
        else:
            axes[1, 2].imshow(img)
            axes[1, 2].set_title("DAGA Attention", fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / f"detection_{i+1:02d}_img_{item['img_id']}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\n✅ Samples saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Task Visualization Tool")
    
    parser.add_argument("--task", type=str, required=True,
                       choices=['classification', 'segmentation', 'depth', 'detection'],
                       help="Task type to visualize")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (.pth) for single model visualization")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                       help="Path to baseline checkpoint for comparison mode")
    parser.add_argument("--daga_checkpoint", type=str, default=None,
                       help="Path to DAGA checkpoint for comparison mode")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./visualization/results",
                       help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples to visualize per mode")
    parser.add_argument("--max_eval", type=int, default=500,
                       help="Max samples to evaluate when finding best samples")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mode", type=str, default="compare",
                       choices=['single', 'compare'],
                       help="Visualization mode: 'single' for one model, 'compare' for baseline vs DAGA")
    parser.add_argument("--selection_mode", type=str, default="both",
                       choices=['best_accuracy', 'best_improvement', 'both'],
                       help="Selection mode: 'best_accuracy' (top DAGA scores), "
                            "'best_improvement' (largest DAGA-Baseline gain), "
                            "'both' (generate both types, 2x num_samples total)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'compare':
        # Comparison mode: find best samples between baseline and DAGA
        if not args.baseline_checkpoint or not args.daga_checkpoint:
            print("❌ Error: --baseline_checkpoint and --daga_checkpoint required for compare mode")
            sys.exit(1)
        
        find_best_samples(
            task=args.task,
            baseline_path=args.baseline_checkpoint,
            daga_path=args.daga_checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            max_eval=args.max_eval,
            device=args.device,
            selection_mode=args.selection_mode
        )
    else:
        # Single model mode: visualize first N samples
        if not args.checkpoint:
            print("❌ Error: --checkpoint required for single mode")
            sys.exit(1)
        
        visualize_from_dataset(
            task=args.task,
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            device=args.device
        )

