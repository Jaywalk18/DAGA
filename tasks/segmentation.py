import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import swanlab
from PIL import Image
import sys

from core.daga import DAGA, process_attention_to_guidance
from core.heads import LinearSegmentationHead
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model

# Add dinov3 to path for official Mask2Former head
sys.path.insert(0, str(Path(__file__).parent.parent / "dinov3"))
try:
    from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
    MASK2FORMER_AVAILABLE = True
except ImportError:
    MASK2FORMER_AVAILABLE = False
    print("[WARN] Mask2FormerHead not available, using LinearHead only")


class SegmentationModel(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=150,
        use_daga=False,
        daga_layers=[11],
        out_indices=[2, 5, 8, 11],
        head_type="linear",  # "linear" or "mask2former"
        hidden_dim=256,  # hidden dim for mask2former
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.out_indices = out_indices
        self.feature_dim = self.vit.embed_dim
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        self.head_type = head_type
        
        self.num_storage_tokens = -1
        self.captured_guidance_attn = None
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if self.use_daga:
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=daga_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
            for param in self.daga.parameters():
                param.requires_grad = True
        
        # Initialize decode head based on type
        embed_dims = [self.feature_dim] * len(out_indices)
        if head_type == "mask2former" and MASK2FORMER_AVAILABLE:
            # Build input_shape dict for Mask2Former
            # Format: {layer_name: (channels, height, width, stride)}
            # We use placeholder values for height/width/stride since they're computed at runtime
            input_shape = {}
            for i, idx in enumerate(out_indices):
                input_shape[str(i + 1)] = (self.feature_dim, None, None, 2 ** (i + 2))
            
            self.decode_head = Mask2FormerHead(
                input_shape=input_shape,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            )
            print(f"✓ Using Mask2FormerHead (hidden_dim={hidden_dim})")
        else:
            self.decode_head = LinearSegmentationHead(
                in_channels=embed_dims,
                num_classes=num_classes
            )
            if head_type == "mask2former":
                print("[WARN] Mask2FormerHead requested but not available, using LinearHead")
            print(f"✓ Using LinearSegmentationHead")
        
        for param in self.decode_head.parameters():
            param.requires_grad = True
        
        print(
            f"✓ SegmentationModel initialized:\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Out indices: {out_indices}\n"
            f"  - Head type: {head_type}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})\n"
            f"  - Guidance layer: {self.daga_guidance_layer_idx}"
        )
    
    def _get_attention_inline(self, block, x):
        """Compute attention weights inline without extra forward"""
        attn_module = block.attn
        normed_x = block.norm1(x)
        
        B, N, C = normed_x.shape
        num_heads = attn_module.num_heads
        head_dim = C // num_heads
        
        qkv = attn_module.qkv(normed_x).reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        return attn
    
    def _get_guidance_map_lightweight(self, x_processed, H, W, num_registers):
        """
        Lightweight forward pass to get last layer's attention (no gradients).
        ~1.5x faster than compute_daga_guidance_map.
        """
        with torch.no_grad():
            features = x_processed.clone()
            for idx, block in enumerate(self.vit.blocks):
                rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                
                if idx == self.daga_guidance_layer_idx:
                    attn_weights = self._get_attention_inline(block, features)
                    guidance_map = process_attention_to_guidance(
                        attn_weights, H, W, num_registers
                    )
                    return guidance_map
                
                features = block(features, rope_sincos)
        
        return None
    
    def forward(self, x, request_visualization_maps=False):
        B = x.shape[0]
        input_size = (x.shape[2], x.shape[3])
        
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        # Stage 1: Get guidance map (optimized lightweight method)
        daga_guidance_map = None
        baseline_attn_weights = None  # Attention before DAGA influence
        adapted_attn_weights = None   # Attention AFTER DAGA modification
        
        if self.use_daga:
            daga_guidance_map = self._get_guidance_map_lightweight(
                x_processed, H, W, num_registers
            )
        
        # Stage 2: Forward with DAGA adaptation
        intermediate_features = []
        
        # Calculate which layer to capture adapted attention
        last_daga_layer = max(self.daga_layers) if self.use_daga and self.daga_layers else -1
        adapted_attn_layer = last_daga_layer if last_daga_layer >= 0 else -1  # Use same layer to capture DAGA effect
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Extract BASELINE attention at guidance layer
            if request_visualization_maps and idx == self.daga_guidance_layer_idx:
                with torch.no_grad():
                    baseline_attn_weights = get_attention_map(block, x_processed)
            
            x_processed = block(x_processed, rope_sincos)
            
            if (
                self.use_daga
                and idx in self.daga_layers
                and daga_guidance_map is not None
            ):
                cls_token = x_processed[:, :1, :]
                register_start_index = 1
                register_end_index = 1 + num_registers
                register_tokens = x_processed[:, register_start_index:register_end_index, :]
                patch_start_index = 1 + num_registers
                patch_tokens = x_processed[:, patch_start_index:, :]
                
                adapted_patch_tokens = self.daga.apply(
                    patch_tokens, daga_guidance_map, idx, H, W
                )
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            # Extract ADAPTED attention at layer after last DAGA layer
            if request_visualization_maps and self.use_daga and idx == adapted_attn_layer:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            if idx in self.out_indices:
                patch_features = x_processed[:, 1 + num_registers:, :]
                feat_spatial = patch_features.transpose(1, 2).reshape(B, C, H, W)
                intermediate_features.append(feat_spatial)
        
        # Decode based on head type
        if self.head_type == "mask2former" and MASK2FORMER_AVAILABLE:
            # Mask2Former expects dict: {layer_name: feature_tensor}
            features_dict = {str(i + 1): feat for i, feat in enumerate(intermediate_features)}
            output = self.decode_head(features_dict)
            # Mask2Former returns dict with 'pred_masks' and 'pred_logits'
            # We need to convert to standard segmentation logits
            pred_masks = output["pred_masks"]  # (B, num_queries, H, W)
            pred_logits = output["pred_logits"]  # (B, num_queries, num_classes+1)
            # Compute semantic segmentation from mask predictions
            mask_cls = pred_logits.softmax(-1)[..., :-1]  # Remove background class
            mask_pred = pred_masks.sigmoid()
            seg_logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
            seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)
        else:
            seg_logits = self.decode_head(intermediate_features, input_size)
        
        # Return adapted attention if available, otherwise baseline
        final_attn = adapted_attn_weights if adapted_attn_weights is not None else baseline_attn_weights
        return seg_logits, final_attn, daga_guidance_map, baseline_attn_weights


def setup_training_components(model, args):
    """Setup criterion, optimizer, scheduler for segmentation"""
    # Use label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)
    
    base_model = model.module if isinstance(model, DataParallel) else model
    
    daga_params = []
    head_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                head_params.append(param)
    
    # Better learning rate scaling
    lr_scaled = args.lr * (args.batch_size * torch.cuda.device_count()) / 16.0
    
    # Use different learning rates for different parts
    param_groups = [{"params": head_params, "lr": lr_scaled, "weight_decay": args.weight_decay}]
    if daga_params:
        param_groups.append(
            {"params": daga_params, "lr": lr_scaled * 0.5, "weight_decay": args.weight_decay * 0.5}
        )
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = 1
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Avoid division by zero for single epoch training
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return criterion, optimizer, scheduler


def calculate_miou(pred, target, num_classes, ignore_index=255):
    """Calculate mean IoU"""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        valid_mask = (target != ignore_index)
        pred_cls = pred_cls & valid_mask
        target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                rank=0, scaler=None, use_amp=False):
    """Train for one epoch with optional AMP support"""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    num_samples = 0
    is_main = (rank == 0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", disable=not is_main)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Call with positional argument to avoid DataParallel kwargs issue
            logits, _, _, _ = model(images, False)
            loss = criterion(logits, masks)
        
        # Backward with scaler if using AMP
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        loss_val = loss.item()
        if torch.isnan(loss) or loss_val != loss_val:
            if is_main:
                print(f"\n⚠️ NaN loss detected at batch {batch_idx}, stopping epoch early")
            return float('inf'), 0.0
        
        total_loss += loss_val
        num_samples += images.size(0)
        
        running_avg_loss = total_loss / (batch_idx + 1)
        
        if is_main:
            pbar.set_postfix({"Loss": f"{running_avg_loss:.4f}"})
    
    return total_loss / max(len(dataloader), 1), 0.0


def evaluate(model, dataloader, device, num_classes):
    """Evaluate segmentation model"""
    model.eval()
    
    total_miou = 0.0
    total_pixel_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Call with positional argument to avoid DataParallel kwargs issue
            logits, _, _, _ = model(images, False)
            preds = logits.argmax(dim=1)
            
            for i in range(images.size(0)):
                miou = calculate_miou(preds[i], masks[i], num_classes)
                total_miou += miou
                
                valid_mask = (masks[i] != 255)
                if valid_mask.sum() > 0:
                    pixel_acc = (preds[i][valid_mask] == masks[i][valid_mask]).float().mean()
                    total_pixel_acc += pixel_acc.item()
                
                num_samples += 1
    
    mean_miou = (total_miou / num_samples * 100) if num_samples > 0 else 0.0
    mean_pixel_acc = (total_pixel_acc / num_samples * 100) if num_samples > 0 else 0.0
    
    return mean_miou, mean_pixel_acc


def visualize_segmentation_results(
    model, fixed_images, fixed_masks, args, output_dir, epoch, colormap=None
):
    """Visualize segmentation predictions with attention maps (separated into two groups)"""
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    # Get actual model from DDP wrapper if needed
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    actual_base_model = base_model.module if isinstance(base_model, (DataParallel, DDP)) else base_model
    
    # Print DAGA status for debugging
    is_daga = getattr(actual_base_model, "use_daga", False)
    if is_daga:
        print(f"\n[DEBUG] Segmentation DAGA Visualization at epoch {epoch+1}:")
        print(f"  DAGA layers: {actual_base_model.daga_layers}")
        print(f"  Visualization layer: {actual_base_model.daga_guidance_layer_idx}")
        print(f"  Spatial scales: {actual_base_model.daga.get_spatial_scales()}")
        print()
    
    with torch.no_grad():
        _, (H, W) = actual_base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        
        logits, adapted_attn_weights, _, baseline_attn_weights = base_model(
            fixed_images, True
        )
        
        preds = logits.argmax(dim=1)
        
        adapted_attn_np = None
        baseline_attn_np = None
        
        # Process adapted attention (captured AFTER DAGA modification)
        if adapted_attn_weights is not None:
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches_expected, H, W)
        
        # Process baseline attention (captured BEFORE DAGA influence)
        if baseline_attn_weights is not None:
            baseline_attn_np = process_attention_weights(baseline_attn_weights, num_patches_expected, H, W)
        
        images_np = fixed_images.cpu().numpy()
        masks_np = fixed_masks.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            # Group 1: Segmentation Results (Image, GT, Prediction)
            fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
            fig1.suptitle(f"Segmentation Results - Epoch {epoch+1} - Sample {j}", fontsize=14, fontweight="bold")
            
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            axes1[0].imshow(img)
            axes1[0].set_title("Original Image")
            axes1[0].axis("off")
            
            # Create a colormap for 150 classes
            # Use nipy_spectral which has better color distribution
            gt_mask_vis = masks_np[j].copy()
            # Mask out ignore regions (255) in black
            gt_mask_display = np.ma.masked_where(gt_mask_vis == 255, gt_mask_vis)
            im1 = axes1[1].imshow(gt_mask_display, cmap='nipy_spectral', vmin=0, vmax=149, interpolation='nearest')
            axes1[1].set_title("Ground Truth")
            axes1[1].axis("off")
            
            # Visualize prediction mask
            pred_mask_vis = preds_np[j].copy()
            pred_mask_vis = np.clip(pred_mask_vis, 0, 149)  # Ensure valid range
            im2 = axes1[2].imshow(pred_mask_vis, cmap='nipy_spectral', vmin=0, vmax=149, interpolation='nearest')
            axes1[2].set_title("Prediction")
            axes1[2].axis("off")
            
            plt.tight_layout()
            vis_figs.append(fig1)
            
            fig1.savefig(
                vis_save_path / f"epoch_{epoch+1}_sample_{j}_segmentation.png",
                dpi=100,
            )
            plt.close(fig1)
            
            # Group 2: Enhanced Attention Map Comparison (only if DAGA is used)
            if adapted_attn_np is not None and baseline_attn_np is not None:
                fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
                fig2.suptitle(f"Attention Analysis - Epoch {epoch+1} - Sample {j}", 
                            fontsize=16, fontweight="bold")
                
                # Row 1: Original attention maps
                im0 = axes2[0, 0].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 0].set_title("Frozen Backbone Attention", fontsize=12)
                axes2[0, 0].axis("off")
                plt.colorbar(im0, ax=axes2[0, 0], fraction=0.046, pad=0.04)
                
                im1 = axes2[0, 1].imshow(adapted_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 1].set_title("DAGA-Adapted Attention", fontsize=12)
                axes2[0, 1].axis("off")
                plt.colorbar(im1, ax=axes2[0, 1], fraction=0.046, pad=0.04)
                
                # Row 2: Difference map and overlay
                diff_map = adapted_attn_np[j] - baseline_attn_np[j]
                abs_diff = np.abs(diff_map)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                
                im2 = axes2[1, 0].imshow(diff_map, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
                axes2[1, 0].set_title(f"Difference Map\n(Mean |Δ|={mean_diff:.4f}, Max |Δ|={max_diff:.4f})", 
                                     fontsize=11)
                axes2[1, 0].axis("off")
                plt.colorbar(im2, ax=axes2[1, 0], fraction=0.046, pad=0.04)
                
                # Overlay difference on image
                axes2[1, 1].imshow(img)
                # Resize attention difference to match image size
                from scipy.ndimage import zoom
                if diff_map.shape != img.shape[:2]:
                    zoom_h = img.shape[0] / diff_map.shape[0]
                    zoom_w = img.shape[1] / diff_map.shape[1]
                    diff_resized = zoom(diff_map, (zoom_h, zoom_w), order=1)
                else:
                    diff_resized = diff_map
                
                im3 = axes2[1, 1].imshow(diff_resized, cmap="RdBu_r", alpha=0.6, vmin=-0.3, vmax=0.3)
                axes2[1, 1].set_title("Difference Overlay on Image", fontsize=11)
                axes2[1, 1].axis("off")
                plt.colorbar(im3, ax=axes2[1, 1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                vis_figs.append(fig2)
                
                fig2.savefig(
                    vis_save_path / f"epoch_{epoch+1}_sample_{j}_attention.png",
                    dpi=100,
                )
                plt.close(fig2)
    
    return vis_figs


def prepare_visualization_data(val_dataset, args, device):
    """Prepare fixed batch for visualization"""
    if not args.enable_visualization:
        return None, None
    
    print(f"\n📸 Preparing visualization data...")
    indices = list(range(min(args.num_vis_samples, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(indices), shuffle=False)
    fixed_images, fixed_masks = next(iter(vis_loader))
    print("✓ Visualization data loaded.")
    return fixed_images.to(device), fixed_masks.to(device)


def run_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    output_dir,
    fixed_vis_images,
    fixed_vis_masks,
    num_classes,
    rank=0,
    world_size=1,
    use_amp=False,
):
    """Execute main training and evaluation loop with optional AMP support"""
    is_main_process = (rank == 0)
    best_miou = 0.0
    val_miou = 0.0  # Initialize val_miou
    start_time = time.time()
    
    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    
    if is_main_process and use_amp:
        print("⚡ Mixed Precision (AMP) enabled")
    
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            rank=rank, scaler=scaler, use_amp=use_amp
        )
        val_miou, val_pixel_acc = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        # Only print on main process
        if is_main_process:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
            print(
                f"   Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.2f}%"
            )
            print(
                f"   Val mIoU: {val_miou:.2f}% | Val Pixel Acc: {val_pixel_acc:.2f}%"
            )
            print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        if is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_miou": val_miou,
                "val_pixel_acc": val_pixel_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_time_minutes": elapsed_time / 60,
            }
            
            if args.enable_visualization and fixed_vis_images is not None and (
                epoch % args.log_freq == 0 or epoch == args.epochs - 1
            ):
                print("📊 Generating segmentation visualizations...")
                vis_figs = visualize_segmentation_results(
                    model, fixed_vis_images, fixed_vis_masks, args, output_dir, epoch
                )
                if vis_figs:
                    log_dict["segmentation_results"] = [
                        swanlab.Image(fig) for fig in vis_figs
                    ]
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
            
            if val_miou > best_miou:
                best_miou = val_miou
                save_path = output_dir / "best_model.pth"
                from core.utils import save_checkpoint
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_miou,
                    args,
                    save_path,
                )
                print(f"   ✅ New best model saved! (Val mIoU: {best_miou:.2f}%)")
    
    return best_miou, val_miou, (time.time() - start_time) / 60
